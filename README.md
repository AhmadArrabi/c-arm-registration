# Lateral Annotation Tool

Quick start guide.

## 1) Install dependencies

From the project root:

```bash
pip install diffdrr matplotlib torch torchvision tqdm
```

This installs the required dependencies explicitly.

## 2) Folder structure

Your project should look like this:

```text
.
├── ccf_lateral_annotations.py
├── pyproject.toml
├── README.md
└── data/
    ├── upper_nifti/
    │   ├── patient_001_BONE_H-N-UXT_3X3.nii.gz
    │   ├── patient_002_BONE_H-N-UXT_3X3.nii.gz
    │   └── ...
    ├── annotations_AP_with_diffdrr.csv
    └── annotations_lateral.csv
```

The script reads:

- `data/annotations_AP_with_diffdrr.csv`
- `data/upper_nifti/*.nii.gz`

and writes:

- `data/annotations_lateral.csv`

## 3) Run it

```bash
python ccf_lateral_annotations.py
```

It will open a DRR window for each patient, starting from the AP annotation pose and rotating it into a lateral view.

## 4) Controls, save, and output

Keyboard controls:

- Left / Right: depth (`diffdrr_y`)
- `+` / `-`: X translation (`diffdrr_x`)
- Up / Down: Z translation (`diffdrr_z`)
- `J` / `L`: LAO / RAO (`diffdrr_a`)
- `I` / `K`: CRA / CAU (`diffdrr_b`)
- `U` / `O`: tilt (`diffdrr_g`)
- `N` / `M`: decrease / increase step size
- `,` / `.`: decrease / increase angle step
- `0`: reset angles to the lateral default

How to save:

- Click the DRR image to save the current pose.
- The script then moves to the next patient.

Output:

- It writes a row into `data/annotations_lateral.csv`.
