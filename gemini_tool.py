import os
import csv
import math
import torch
import matplotlib.pyplot as plt
import diffdrr.data as diffdrr_data
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform, make_matrix

def render_drrs_in_chunks(poses, drr, drr_batch_size, transform=None):
    """Helper function to render a large batch of poses in memory-safe chunks."""
    drr_chunks = []
    for j in range(0, len(poses), drr_batch_size):
        chunk_matrix = poses.matrix[j : j + drr_batch_size]
        chunk_poses = RigidTransform(chunk_matrix)
        chunk_drrs = drr(chunk_poses)
        drr_chunks.append(chunk_drrs)

    drr_chunks = torch.cat(drr_chunks, dim=0)
    drr_chunks = torch.flip(drr_chunks, dims=[3])
    if transform is not None:
        drr_chunks = transform(drr_chunks)
    return drr_chunks

def append_to_csv(csv_path, sample_file, idx, x_val, y_val, z_val, pose_matrix):
    """Saves the clicked pose and sample data to a CSV file."""
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            headers = ['sample_file', 'clicked_index', 'x_translation', 'y_translation', 'z_translation']
            headers.extend([f'm{i}{j}' for i in range(4) for j in range(4)])
            writer.writerow(headers)
        
        flat_matrix = pose_matrix.flatten().tolist()
        row = [sample_file, idx, float(x_val), float(y_val), float(z_val)] + flat_matrix
        writer.writerow(row)
        
    print(f"Saved annotation to {csv_path}")


class InteractiveDRRAnnotator:
    def __init__(self, drr_module, sample_file, csv_path, sdd=1020., B=2):
        self.drr = drr_module
        self.sample_file = sample_file
        self.csv_path = csv_path
        self.device = drr_module.device
        self.B = B
        self.N = B * B
        self.drr_batch_size = 8
        
        # Grid settings
        self.grid_extent = 50.0  # Defines spread of the grid from the center
        self.move_step = 20.0    # How much arrows move the center
        self.zoom_step = 20.0    # How much +/- moves the Y axis
        
        # State tracking
        self.center_x = 0.0
        self.center_y = -(sdd - 300.0)
        self.center_z = 0.0
        self.current_poses = None
        self.current_offsets = None
        self.hovered_ax = None
        
        # Base Rotation
        theta = 180. * math.pi / 180.0
        self.R = torch.tensor([[[math.cos(theta), -math.sin(theta), 0.0], 
                                [math.sin(theta), math.cos(theta), 0.0], 
                                [0.0, 0.0, 1.0]]], device=self.device)

        # Matplotlib UI setup
        self.fig, self.axes = plt.subplots(self.B, self.B, figsize=(10, 10))
        self.fig.canvas.manager.set_window_title('Interactive DRR Annotator')
        self.axes = [self.axes] if self.B == 1 else self.axes.flatten()
        
        self.img_plots = []
        for ax in self.axes:
            ax.axis('off')
            # Initialize with empty images
            img_plot = ax.imshow(torch.zeros(256, 256).numpy(), cmap='gray', vmin=0, vmax=1)
            self.img_plots.append(img_plot)
            
        # Connect events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)
        
        self.update_view()

    def generate_current_drrs(self):
        """Generates the DRR grid based on current center coordinates."""
        # Generate Grid Steps relative to current center
        x_steps = torch.linspace(self.center_x - self.grid_extent, self.center_x + self.grid_extent, self.B, device=self.device)
        z_steps = torch.linspace(self.center_z - self.grid_extent, self.center_z + self.grid_extent, self.B, device=self.device)
        
        X, Z = torch.meshgrid(x_steps, z_steps, indexing='ij')
        Y = torch.full_like(X, self.center_y)
        
        self.current_offsets = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)
        
        R_batch = self.R.repeat(self.N, 1, 1)
        # We apply offsets directly as the entire translation vector
        batch_poses = RigidTransform(make_matrix(R_batch, self.current_offsets))
        self.current_poses = batch_poses

        # Render
        drr_images = render_drrs_in_chunks(batch_poses, self.drr, self.drr_batch_size)
        
        # Normalize for display globally across the batch
        drr_images = drr_images - drr_images.min()
        drr_images = drr_images / (drr_images.max() + 1e-8)
        
        return drr_images

    def update_view(self):
        """Forces UI update, renders new DRRs, and pushes them to Matplotlib."""
        self.fig.suptitle('Rendering new poses... Please wait.', color='red', fontsize=14)
        self.fig.canvas.draw_idle()
        plt.pause(0.01) # Force UI flush
        
        drr_images = self.generate_current_drrs()
        
        for i, ax in enumerate(self.axes):
            img = drr_images[i].squeeze().cpu().numpy()
            self.img_plots[i].set_data(img)
            
            x_val = self.current_offsets[i, 0].item()
            z_val = self.current_offsets[i, 2].item()
            ax.set_title(f"X: {x_val:.1f} | Z: {z_val:.1f}")
            
        self.fig.suptitle(f"Controls: Arrows (Move X/Z) | +/- (Zoom/Y)\nCenter: X={self.center_x:.1f}, Y={self.center_y:.1f}, Z={self.center_z:.1f}", color='black', fontsize=12)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        """Handle keyboard navigation."""
        needs_update = False
        
        if event.key == 'up':
            self.center_z += self.move_step
            needs_update = True
        elif event.key == 'down':
            self.center_z -= self.move_step
            needs_update = True
        elif event.key == 'right':
            self.center_x += self.move_step
            needs_update = True
        elif event.key == 'left':
            self.center_x -= self.move_step
            needs_update = True
        elif event.key in ['+', '=']: # Zoom In (Move source closer)
            self.center_y += self.zoom_step
            needs_update = True
        elif event.key == '-': # Zoom Out (Move source further)
            self.center_y -= self.zoom_step
            needs_update = True
            
        if needs_update:
            self.update_view()

    def on_hover(self, event):
        """Highlights the image currently under the mouse cursor."""
        if event.inaxes is not None:
            if self.hovered_ax != event.inaxes:
                # Clear old hover
                if self.hovered_ax is not None:
                    for spine in self.hovered_ax.spines.values():
                        spine.set_edgecolor('white')
                        spine.set_linewidth(0)
                        
                # Set new hover
                self.hovered_ax = event.inaxes
                for spine in self.hovered_ax.spines.values():
                    self.hovered_ax.axis('on')
                    spine.set_edgecolor('red')
                    spine.set_linewidth(4)
                self.fig.canvas.draw_idle()
        else:
            # Mouse left axes, clear all
            if self.hovered_ax is not None:
                for spine in self.hovered_ax.spines.values():
                    spine.set_edgecolor('white')
                    spine.set_linewidth(0)
                self.hovered_ax.axis('off')
                self.hovered_ax = None
                self.fig.canvas.draw_idle()

    def on_click(self, event):
        """Save the selected pose and exit."""
        if event.inaxes is not None:
            for idx, ax in enumerate(self.axes):
                if ax == event.inaxes:
                    x_val = self.current_offsets[idx, 0].item()
                    y_val = self.current_offsets[idx, 1].item()
                    z_val = self.current_offsets[idx, 2].item()
                    pose_matrix = self.current_poses.matrix[idx].cpu().numpy()
                    
                    append_to_csv(self.csv_path, self.sample_file, idx, x_val, y_val, z_val, pose_matrix)
                    
                    plt.close(self.fig)
                    break

def main():
    sample_file = "Y:/biplane_positioning/data/CTA/nifti/104006334-1.2.840.113696.376376.500.37870131.20170220102913/4_cta_carotid__075__b20f.nii.gz"
    csv_output_file = "pose_annotations.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading volume...")
    subject = diffdrr_data.read(sample_file, bone_attenuation_multiplier=5)
    sdd = 1020.
    drr = DRR(subject, sdd=sdd, height=256, delx=1).to(device)
    
    print("Opening interactive annotator...")
    
    # Initialize the annotator loop (B=2 makes a 2x2 grid, easily change to B=3 for a 3x3)
    annotator = InteractiveDRRAnnotator(
        drr_module=drr,
        sample_file=sample_file,
        csv_path=csv_output_file,
        sdd=sdd,
        B=1
    )
    
    # Starts the blocking UI event loop
    plt.show()

if __name__ == "__main__":
    main()