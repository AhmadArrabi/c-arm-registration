import diffdrr.data as diffdrr_data
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform, make_matrix
import torch
from diffdrr.visualization import visualize_scene
import math
import torchvision

def render_drrs_in_chunks(poses, drr, sid, drr_batch_size, transform=None, max_sdd=None):
    """Helper function to render a large batch of poses in memory-safe chunks."""
    drr_chunks = []
    for j in range(0, len(poses), drr_batch_size):
        chunk_poses = RigidTransform(poses[j : j + drr_batch_size])
        chunk_drrs = drr(chunk_poses)
        drr_chunks.append(chunk_drrs)

    drr_chunks = torch.cat(drr_chunks, dim=0)
    drr_chunks = torch.flip(drr_chunks, dims=[3])
    if transform is not None:
        drr_chunks = transform(drr_chunks)
    return drr_chunks

def main():
    sample_file = "./data/case-100012_BONE_H-N-UXT_3X3.nii.gz"
    subject = diffdrr_data.read(sample_file, bone_attenuation_multiplier=5)
    drr = DRR(subject, sdd=1020., height=256, delx=1)
    B = 2
    N = B * B  # Total number of poses
    sdd = 1020.  # Source-to-Detector Distance

    theta = 180. * math.pi / 180.0
    #make rotation matrix
    R = torch.tensor([[[math.cos(theta), -math.sin(theta), 0.0], 
                      [math.sin(theta), math.cos(theta), 0.0], 
                      [0.0, 0.0, 1.0]]])
    translation = torch.tensor([[0.0, - (sdd - 300.), 0.0]])
    
    grid_extent = 50.0 
    x_steps = torch.linspace(-grid_extent, grid_extent, B)
    z_steps = torch.linspace(-grid_extent, grid_extent, B)
    
    X, Z = torch.meshgrid(x_steps, z_steps, indexing='ij')
    Y = torch.zeros_like(X) # No movement in Y
    
    offsets = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)
    
    R_batch = R.repeat(N, 1, 1)
    translation_batch = translation.repeat(N, 1) + offsets
    
    batch_poses = RigidTransform(
        make_matrix(
            R_batch,
            translation_batch,
        )
    )

    drr_batch_size = 2 # Adjust based on your GPU VRAM
    print(f"Rendering {N} DRRs in chunks of {drr_batch_size}...")
    
    drr_images = render_drrs_in_chunks(
        poses=batch_poses, 
        drr=drr, 
        sid=sdd, 
        drr_batch_size=drr_batch_size
    )

    output_path = "drr_grid.png"
    torchvision.utils.save_image(
        drr_images, 
        output_path, 
        nrow=B,          # Number of images displayed in each row of the grid
        normalize=True,  # Scales tensors to [0, 1] for visualization
        padding=2        # Adds a 2-pixel border between grid elements
    )
    print(f"Saved {B}x{B} grid to {output_path}")


if __name__ == "__main__":
    main()
