import numpy as np
from scipy.spatial.transform import Rotation
import pandas as pd
from scipy.spatial.transform import Rotation as R
import open3d as o3d


def generate_rand_clouds(
    n_points,
    std_noise=1e-2,
    max_scale=10,
    max_translation=10,
    n_random_outliers=0,
    save_path="./CSV",
):
    # Generate source point cloud
    src = np.random.rand(3, n_points)

    # Add noise to create the target point cloud
    dst = src + np.random.normal(size=(3, n_points), scale=std_noise)

    # Identify outliers
    outliers = np.arange(n_points)
    np.random.shuffle(outliers)
    outliers = outliers[:n_random_outliers]

    # Assign random values to outliers
    dst[:, outliers] = np.random.rand(3, n_random_outliers)

    # Pick a transformation
    scale = np.random.uniform(1, max_scale)
    translation = np.random.rand(3, 1) * max_translation
    rotation = Rotation.random().as_matrix()

    # Apply the transformation
    dst = scale * np.matmul(rotation, dst) + translation

    # Create a DataFrame for the target cloud with an "inlier" column
    inlier_labels = np.ones(n_points, dtype=int)  # 1 for inliers
    inlier_labels[outliers] = 0  # 0 for outliers

    df_dst = pd.DataFrame(
        {"x": dst[0, :], "y": dst[1, :], "z": dst[2, :], "inlier": inlier_labels}
    )

    # Save target cloud with inlier/outlier labels
    cloud_filename = f"{save_path}/target_cloud.csv"
    df_dst.to_csv(cloud_filename, index=False)

    # Save transformation parameters
    transformation_filename = f"{save_path}/transformation_params.csv"
    df_transformation = pd.DataFrame(
        {
            "scale": [scale],
            "translation_x": [translation[0, 0]],
            "translation_y": [translation[1, 0]],
            "translation_z": [translation[2, 0]],
            "rotation_matrix": [
                rotation.flatten().tolist()
            ],  # Flatten rotation matrix for easy saving
        }
    )
    df_transformation.to_csv(transformation_filename, index=False)

    print(f"Target cloud saved to: {cloud_filename}")
    print(f"Transformation parameters saved to: {transformation_filename}")

    return src, dst, scale, translation, rotation


def generate_transformed_cloud(
    A, max_s=5, max_t=1, Rot=True, outlier_ratio=0.0, noise_std=0.01
):

    s_gt = np.random.uniform(1, max_s)  # Random scale in [1,5]
    if Rot == True:
        R_gt = R.random().as_matrix()
    else:
        R_gt = np.eye(3)

    t_gt = np.random.uniform(-max_t, max_t, size=(3,))

    B_clean = s_gt * (A @ R_gt.T) + t_gt
    noise = np.random.normal(0, noise_std, B_clean.shape)
    B = B_clean + noise

    N = A.shape[0]
    num_outliers = int(outlier_ratio * N)
    out_mask = np.zeros(N, dtype=bool)
    if num_outliers > 0:
        outlier_indices = np.random.choice(N, num_outliers, replace=False)
        out_mask[outlier_indices] = True
        # Replace the selected points with random points in [-5,5]^3
        B[outlier_indices] = np.random.uniform(-5, 5, size=(num_outliers, 3))
    return B, s_gt, R_gt, t_gt, out_mask


def load_bunny(N=1000):
    PLY_FILE = "bunny.ply"
    bunny_pcd = o3d.io.read_point_cloud(PLY_FILE)
    # Downsample roughly to 1000 points (adjust voxel size as needed)
    downsampled_pcd = bunny_pcd.voxel_down_sample(voxel_size=0.005)
    A = np.asarray(downsampled_pcd.points)

    print(A.shape)
    if A.shape[0] > N:
        idx = np.random.choice(A.shape[0], N, replace=False)
        A = A[idx, :]
    # Normalize
    A = (A - np.min(A, axis=0)) / (np.max(A, axis=0) - np.min(A, axis=0))
    print(f"Point cloud has {A.shape[0]} points.")

    # Create a new point cloud to visualize the processed data
    processed_pcd = o3d.geometry.PointCloud()
    processed_pcd.points = o3d.utility.Vector3dVector(A)
    o3d.visualization.draw_geometries([processed_pcd], window_name="Downsampled Bunny")

    return A
