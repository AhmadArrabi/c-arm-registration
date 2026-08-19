"""
Lateral skull DRR annotation tool.

Reads AP annotations from data/annotations_AP_with_diffdrr.csv, initialises
the camera rotated 90 degrees around the LAO/RAO axis so that the skull is
viewed from the side (lateral), then lets the annotator fine-tune the pose
and save it to data/annotations_lateral.csv.

The output CSV uses the same column schema as the AP annotation pipeline so
it can be fed directly into the training code with no changes.

Controls (keyboard)
-------------------
Left / Right : depth (diffdrr_y)
+ / -        : X translation (diffdrr_x)
Up / Down    : Z translation (diffdrr_z)
J / L        : LAO / RAO (diffdrr_a)
I / K        : CRA / CAU (diffdrr_b)
U / O        : Tilt (diffdrr_g)
N / M        : decrease / increase translation step
, / .        : decrease / increase angle step
0            : reset angles to current lateral defaults (keep x,y,z)
Click DRR    : save current pose and move to next patient

Usage
-----
    python src/ccf_lateral_annotation.py

Adjust the constants in main() to point at your data directory and desired
output CSV path.
"""

import csv
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from diffdrr.drr import DRR
import diffdrr.data as diffdrr_data
from diffdrr.pose import RigidTransform, make_matrix


OUTLIER_PATIENT_IDS = {"case-105908", "case-102367"}


# ── suppress matplotlib default key bindings that interfere with our controls ──
for _keymap in [
    "keymap.save",
    "keymap.quit",
    "keymap.quit_all",
    "keymap.grid",
    "keymap.grid_minor",
    "keymap.fullscreen",
    "keymap.home",
    "keymap.back",
    "keymap.forward",
    "keymap.pan",
    "keymap.zoom",
    "keymap.xscale",
    "keymap.yscale",
]:
    mpl.rcParams[_keymap] = []


# ─────────────────────────────── DRR utilities ────────────────────────────────

def build_pose_from_carm(carm_params, sdd=1020.0, sid=700.0):
    """
    Build a RigidTransform pose from C-arm parameters [x, y, z, a, b, g].

    Parameter order matches the training codebase convention:
        x, y, z : isocenter position (mm)
        a        : LAO/RAO rotation (deg)
        b        : CRA/CAU rotation (deg)
        g        : roll / tilt (deg)
    """
    _ = sdd  # kept for config parity with training code

    device = carm_params.device
    iso = carm_params[:, :3]
    sid_value = float(sid)

    alpha_deg = carm_params[:, 3]
    beta_deg  = carm_params[:, 4]
    gamma_deg = carm_params[:, 5]

    # Place source at isocenter + SID along Y, then rotate
    source0 = iso + torch.tensor([0.0, sid_value, 0.0], device=device, dtype=torch.float32)
    relative = source0 - iso

    alpha = torch.deg2rad(alpha_deg)
    beta  = torch.deg2rad(beta_deg)
    gamma = torch.deg2rad(gamma_deg)
    zero  = torch.zeros_like(alpha)
    one   = torch.ones_like(alpha)

    # LAO/RAO rotation around Z
    R_lao = torch.stack([
        torch.stack([torch.cos(alpha), -torch.sin(alpha), zero], dim=-1),
        torch.stack([torch.sin(alpha),  torch.cos(alpha), zero], dim=-1),
        torch.stack([zero,              zero,              one ], dim=-1),
    ], dim=-2)

    # CRA/CAU rotation around X
    R_cra = torch.stack([
        torch.stack([one,  zero,             zero            ], dim=-1),
        torch.stack([zero, torch.cos(beta), -torch.sin(beta) ], dim=-1),
        torch.stack([zero, torch.sin(beta),  torch.cos(beta) ], dim=-1),
    ], dim=-2)

    # Roll rotation around Y
    R_roll = torch.stack([
        torch.stack([ torch.cos(gamma), zero, torch.sin(gamma)], dim=-1),
        torch.stack([zero,              one,  zero             ], dim=-1),
        torch.stack([-torch.sin(gamma), zero, torch.cos(gamma) ], dim=-1),
    ], dim=-2)

    R = R_lao @ R_cra @ R_roll
    source = iso + torch.matmul(R, relative.unsqueeze(-1)).squeeze(-1)

    return RigidTransform(make_matrix(R, source))


def render_drrs_in_chunks(carm_params, drr, sdd, sid, drr_batch_size):
    """Render a batch of C-arm parameters as DRRs in memory-safe chunks."""
    chunks = []
    for j in range(0, carm_params.size(0), drr_batch_size):
        chunk = carm_params[j : j + drr_batch_size]
        pose  = build_pose_from_carm(chunk, sdd=sdd, sid=sid)
        chunks.append(drr(pose))
    return torch.flip(torch.cat(chunks, dim=0), dims=[3])


# ─────────────────────────────── CSV utilities ────────────────────────────────

def _lateral_csv_headers():
    """Column schema matching the AP annotation pipeline standard."""
    headers = [
        "sample_file",
        "clicked_index",
        "diffdrr_x",
        "diffdrr_y",
        "diffdrr_z",
        "diffdrr_a",
        "diffdrr_b",
        "diffdrr_g",
        "lao_rao",
        "cra_cau",
    ]
    headers.extend([f"m{i}{j}" for i in range(4) for j in range(4)])
    return headers


def load_ap_pose(ap_csv_path, patient_id):
    """
    Return [diffdrr_x, y, z, a, b, g] for *patient_id* from the AP CSV.

    Supports CSVs that have either:
      • ``diffdrr_x/y/z/a/b/g`` columns  (annotations_AP_with_diffdrr*.csv)
      • plain ``x/y/z/a/b/g`` columns    (annotations_AP.csv)
    """
    ap_csv_path = Path(ap_csv_path)
    if not ap_csv_path.is_file():
        return None
    with open(ap_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("patient") == patient_id:
                if "diffdrr_x" in row:
                    return [
                        float(row["diffdrr_x"]),
                        float(row["diffdrr_y"]),
                        float(row["diffdrr_z"]),
                        float(row["diffdrr_a"]),
                        float(row["diffdrr_b"]),
                        float(row["diffdrr_g"]),
                    ]
                else:
                    # fall back to raw columns (not in diffdrr coord frame, but
                    # still usable as a coarse initialisation)
                    return [
                        float(row["x"]),
                        float(row["y"]),
                        float(row["z"]),
                        float(row["a"]),
                        float(row["b"]),
                        float(row["g"]),
                    ]
    return None


def load_lateral_pose(lateral_csv_path, sample_file):
    """Return [diffdrr_x, y, z, a, b, g] from the lateral CSV, or None."""
    lateral_csv_path = Path(lateral_csv_path)
    if not lateral_csv_path.is_file():
        return None
    with open(lateral_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("sample_file") == sample_file:
                return [
                    float(row["diffdrr_x"]),
                    float(row["diffdrr_y"]),
                    float(row["diffdrr_z"]),
                    float(row["diffdrr_a"]),
                    float(row["diffdrr_b"]),
                    float(row["diffdrr_g"]),
                ]
    return None


def upsert_lateral_csv(csv_path, sample_file, idx, params_6, pose_matrix):
    """Write pose to the lateral CSV, replacing any existing row for sample_file."""
    csv_path = Path(csv_path)
    x, y, z, a, b, g = [float(v) for v in params_6]
    lao_rao = a - 180.0
    cra_cau = -b
    flat_matrix = pose_matrix.flatten().tolist()

    new_row = [sample_file, int(idx), x, y, z, a, b, g, lao_rao, cra_cau, *flat_matrix]
    headers = _lateral_csv_headers()

    existing_rows = []
    if csv_path.is_file():
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            file_headers = next(reader, None)
            if file_headers is not None:
                headers = file_headers
            for row in reader:
                if row and row[0] != sample_file:
                    existing_rows.append(row)

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(existing_rows)
        writer.writerow(new_row)

    print(
        "Saved lateral annotation:",
        {
            "csv": str(csv_path),
            "diffdrr": {"x": x, "y": y, "z": z, "a": a, "b": b, "g": g},
            "clinical approx": {"lao_rao": lao_rao, "cra_cau": cra_cau},
        },
    )


# ──────────────────────────── Interactive annotator ───────────────────────────

class LateralDRRAnnotator:
    """
    Single-panel interactive DRR annotator for lateral skull views.

    Initialisation
    ~~~~~~~~~~~~~~
    If a prior lateral annotation exists for this patient it is loaded directly.
    Otherwise the AP pose is rotated 90 degrees around the LAO/RAO axis
    (``a -= 90``) to give a good lateral starting point.

    Saving
    ~~~~~~
    Click the DRR image to save the current pose and close the window,
    advancing to the next patient.
    """

    def __init__(
        self,
        drr_module,
        sample_file,
        lateral_csv_path,
        patient_id,
        ap_pose,            # [x, y, z, a, b, g] from AP annotation
        sdd=1020.0,
        sid=700.0,
    ):
        self.drr            = drr_module
        self.sample_file    = sample_file
        self.lateral_csv    = Path(lateral_csv_path)
        self.patient_id     = patient_id
        self.sdd            = float(sdd)
        self.sid            = float(sid)
        self.device         = drr_module.device

        self.drr_batch_size = 1  # single pose at a time for the annotator
        self.move_step      = 10.0
        self.zoom_step      = 10.0
        self.angle_step     = 1.0

        # ── Determine initial pose ──────────────────────────────────────────
        prior_lateral = load_lateral_pose(self.lateral_csv, sample_file)
        if prior_lateral is not None:
            cx, cy, cz, ca, cb, cg = prior_lateral
            print(f"  Resuming prior lateral pose: {prior_lateral}")
        else:
            # Rotate AP 90° around LAO/RAO axis → lateral view
            cx, cy, cz, ca_ap, cb_ap, cg_ap = ap_pose
            ca = ca_ap - 90.0   # e.g. 180 → 90  (left lateral)
            cb = 0.0
            cg = 0.0
            print(f"  Initialising from AP pose, lateral a={ca:.1f}")

        self.center_x = cx
        self.center_y = cy
        self.center_z = cz
        self.center_a = ca
        self.center_b = cb
        self.center_g = cg

        # Defaults used by the '0' reset key
        self._default_a = ca
        self._default_b = 0.0
        self._default_g = 0.0

        self.current_params = None
        self.current_poses  = None
        self.latest_drr_img = None

        # ── Build UI ────────────────────────────────────────────────────────
        self.fig, self.ax = plt.subplots(1, 1, figsize=(9, 8))
        self.fig.canvas.manager.set_window_title(
            f"Lateral Annotator – {patient_id}"
        )
        self.ax.axis("off")
        self.drr_plot = self.ax.imshow(
            torch.zeros(256, 256).numpy(), cmap="gray", vmin=0, vmax=1
        )

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        self.update_view()

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _make_params(self):
        return torch.tensor(
            [[self.center_x, self.center_y, self.center_z,
              self.center_a, self.center_b, self.center_g]],
            dtype=torch.float32,
            device=self.device,
        )

    def generate_drr(self):
        self.current_params = self._make_params()
        self.current_poses  = build_pose_from_carm(
            self.current_params, sdd=self.sdd, sid=self.sid
        )
        imgs = render_drrs_in_chunks(
            self.current_params, self.drr,
            sdd=self.sdd, sid=self.sid,
            drr_batch_size=self.drr_batch_size,
        )
        imgs = imgs - imgs.min()
        imgs = imgs / (imgs.max() + 1e-8)
        self.latest_drr_img = imgs[0].squeeze().detach().cpu().numpy()

    def _status_text(self):
        lao_rao = self.center_a - 180.0
        cra_cau = -self.center_b
        return (
            f"Patient: {self.patient_id}   |   Click DRR image to SAVE\n"
            "Controls: Left/Right depth(Y) | +/- X | Up/Down Z | "
            "J/L LAO/RAO | I/K CRA/CAU | U/O Tilt | N/M step | ,/. angle | 0 reset\n"
            f"x={self.center_x:.1f}  y={self.center_y:.1f}  z={self.center_z:.1f}  "
            f"lao_rao={self.center_a:.1f}  cra_cau={self.center_b:.1f}  tilt={self.center_g:.1f}  "
            f"|  LAO/RAO={lao_rao:.1f}  CRA/CAU={cra_cau:.1f}  "
            f"|  step={self.move_step:.1f}mm / {self.angle_step:.1f}°"
        )

    def update_view(self):
        self.fig.suptitle("Rendering… please wait.", color="red", fontsize=11)
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

        self.generate_drr()
        self.drr_plot.set_data(self.latest_drr_img)
        self.ax.set_title(
            f"Lateral DRR  —  LAO/RAO={self.center_a:.1f}  CRA/CAU={self.center_b:.1f}  "
            f"Tilt={self.center_g:.1f}",
            fontsize=10,
        )
        self.fig.suptitle(self._status_text(), color="black", fontsize=9)
        self.fig.canvas.draw_idle()

    # ── Keyboard controls ──────────────────────────────────────────────────────

    def on_key(self, event):
        key = event.key
        needs_update = False

        # Left/right now control depth (Y axis).
        if key == "left":
            self.center_y -= self.zoom_step
            needs_update = True
        elif key == "right":
            self.center_y += self.zoom_step
            needs_update = True
        # +/- now control the other translation axis (X axis).
        elif key in ["+", "="]:
            self.center_x += self.move_step
            needs_update = True
        elif key == "-":
            self.center_x -= self.move_step
            needs_update = True
        elif key == "up":
            self.center_z += self.move_step
            needs_update = True
        elif key == "down":
            self.center_z -= self.move_step
            needs_update = True
        elif key == "j":
            self.center_a -= self.angle_step
            needs_update = True
        elif key == "l":
            self.center_a += self.angle_step
            needs_update = True
        elif key == "i":
            self.center_b += self.angle_step
            needs_update = True
        elif key == "k":
            self.center_b -= self.angle_step
            needs_update = True
        elif key == "u":
            self.center_g -= self.angle_step
            needs_update = True
        elif key == "o":
            self.center_g += self.angle_step
            needs_update = True
        elif key == "n":
            self.move_step = max(1.0, self.move_step - 1.0)
            self.zoom_step = max(1.0, self.zoom_step - 1.0)
            needs_update = True
        elif key == "m":
            self.move_step += 1.0
            self.zoom_step += 1.0
            needs_update = True
        elif key == ",":
            self.angle_step = max(0.1, round(self.angle_step - 0.1, 2))
            needs_update = True
        elif key == ".":
            self.angle_step = round(self.angle_step + 0.1, 2)
            needs_update = True
        elif key == "0":
            self.center_a = self._default_a
            self.center_b = self._default_b
            self.center_g = self._default_g
            needs_update = True

        if needs_update:
            self.update_view()

    # ── Mouse click to save ───────────────────────────────────────────────────

    def on_click(self, event):
        if event.inaxes is None or event.inaxes != self.ax:
            return

        params = self.current_params[0].detach().cpu().numpy()
        pose_matrix = self.current_poses.matrix[0].detach().cpu().numpy()

        upsert_lateral_csv(
            self.lateral_csv,
            self.sample_file,
            0,
            params,
            pose_matrix,
        )
        plt.close(self.fig)


# ──────────────────────────────────── main ────────────────────────────────────

def main():
    # ── Configuration ─────────────────────────────────────────────────────────
    nifti_dir         = "data/upper_nifti"      # directory with .nii.gz files
    ap_csv_path       = "data/annotations_AP_with_diffdrr.csv"  # AP annotations
    lateral_csv_out   = "data/annotations_lateral.csv"          # output

    sdd   = 1020.0   # source-to-detector distance (mm)
    sid   = 700.0    # source-to-isocenter distance (mm)
    drr_size = 256   # image size in pixels
    bone_attenuation_multiplier = 5.0

    # ── Parse AP CSV to get patient list ──────────────────────────────────────
    ap_csv_path  = Path(ap_csv_path)
    nifti_dir    = Path(nifti_dir)
    lateral_csv  = Path(lateral_csv_out)

    if not ap_csv_path.is_file():
        raise FileNotFoundError(f"AP annotations CSV not found: {ap_csv_path}")

    patients = []
    seen = set()
    skipped_outliers = 0
    skipped_duplicates = 0
    with open(ap_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("patient") or row.get("sample_file")
            if not pid:
                continue
            if pid in OUTLIER_PATIENT_IDS:
                skipped_outliers += 1
                continue
            if pid in seen:
                skipped_duplicates += 1
                continue
            seen.add(pid)
            patients.append(pid)

    if not patients:
        raise ValueError("No patient rows found in AP CSV.")

    print(f"Found {len(patients)} patient(s) in AP CSV after filtering outliers.")
    print(
        f"Skipped {skipped_outliers} outlier row(s) and {skipped_duplicates} duplicate row(s)."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Iterate over patients ─────────────────────────────────────────────────
    for sample_idx, patient_id in enumerate(patients, start=1):
        nifti_path = nifti_dir / f"{patient_id}_BONE_H-N-UXT_3X3.nii.gz"

        if not nifti_path.is_file():
            print(f"[{sample_idx}/{len(patients)}] SKIP – nifti not found: {nifti_path}")
            continue

        # Load AP pose to seed lateral initialisation
        ap_pose = load_ap_pose(ap_csv_path, patient_id)
        if ap_pose is None:
            print(f"[{sample_idx}/{len(patients)}] SKIP – no AP annotation for {patient_id}")
            continue

        print(f"\n[{sample_idx}/{len(patients)}] Patient: {patient_id}")
        print(f"  AP pose (diffdrr): {ap_pose}")
        print(f"  Loading volume: {nifti_path}")

        subject = diffdrr_data.read(
            str(nifti_path),
            bone_attenuation_multiplier=bone_attenuation_multiplier,
        )
        drr = DRR(subject, sdd=sdd, height=drr_size, delx=1).to(device)

        print("  Opening lateral annotator…")
        annotator = LateralDRRAnnotator(
            drr_module       = drr,
            sample_file      = str(nifti_path),
            lateral_csv_path = lateral_csv,
            patient_id       = patient_id,
            ap_pose          = ap_pose,
            sdd              = sdd,
            sid              = sid,
        )
        _ = annotator  # keeps reference alive until plt.show() returns
        plt.show()

        # Free GPU memory before loading next volume
        del drr, subject
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nFinished. Annotations saved to: {lateral_csv}")


if __name__ == "__main__":
    main()
