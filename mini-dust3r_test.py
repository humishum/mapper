import argparse
from pathlib import Path
import numpy as np
import torch

# Mini-DUSt3R API
from mini_dust3r.api import inferece_dust3r
from mini_dust3r.model import AsymmetricCroCo3DStereo
from mini_dust3r.utils.image import load_images

# Open3D for point-cloud handling and visualization
import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct and visualize a 3D scene from images using Mini-DUSt3R"
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        required=True,
        help="Path to folder containing input images (.png, .jpg, .jpeg)"
    )
    parser.add_argument(
        "--output_ply",
        default="scene.ply",
        help="Filename for exporting the merged point cloud"
    )
    parser.add_argument(
        "--niter",
        type=int,
        default=100,
        help="Number of optimization iterations"
    )
    return parser.parse_args()


def main(image_dir: Path, output_ply: str, niter: int):
    # Select device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load pretrained model
    model = AsymmetricCroCo3DStereo.from_pretrained(
        "nielsr/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    ).to(device)

    # Load images
    rgb_list = load_images(str(image_dir), size=512)
    if not rgb_list:
        print(f"No images found in {image_dir}")
        return

    # Perform inference (builds a single optimized point cloud)
    optimized_results = inferece_dust3r(
        image_dir_or_list=image_dir,
        model=model,
        device=device,
        batch_size=1,
        niter=niter,
    )

    # Extract trimesh PointCloud and convert to numpy
    trimesh_pcd = optimized_results.point_cloud
    points = np.asarray(trimesh_pcd.vertices)
    
    # Extract colors if available
    colors = None
    if hasattr(trimesh_pcd, 'colors') and trimesh_pcd.colors is not None:
        colors = np.asarray(trimesh_pcd.colors)
        print(f"Found colors with shape: {colors.shape}")
        # Convert from RGBA (0-255) to RGB (0-1) if needed
        if colors.shape[1] == 4:  # RGBA
            colors = colors[:, :3]  # Take only RGB
        if colors.max() > 1.0:  # Convert from 0-255 to 0-1 range
            colors = colors.astype(np.float64) / 255.0
    else:
        print("No colors found in the point cloud")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Add colors if available
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print("Applied colors to Open3D point cloud")

    # Save to PLY
    o3d.io.write_point_cloud(output_ply, pcd)
    print(f"Saved merged point cloud to {output_ply}")

    # Visualize
    print("Launching Open3D visualizer...")
    o3d.visualization.draw_geometries([pcd])

    # Debugging output
    print("Available attributes:", dir(trimesh_pcd))

if __name__ == "__main__":
    args = parse_args()
    main(args.image_dir, args.output_ply, args.niter)