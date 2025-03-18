import numpy as np
import matplotlib.pyplot as plt
import teaserpp_python
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from sklearn.linear_model import RANSACRegressor

from data import generate_transformed_cloud, load_bunny

# -------------------------
# 1) Load and normalize Bunny point cloud
# -------------------------
A = load_bunny(N=40)

# -------------------------
# 2) Monte Carlo Simulation Parameters for Rotation Estimation
# -------------------------
num_trials = 10
outlier_rates = np.arange(1, 100)  # Testing extreme outlier cases

# Store rotation errors for each method
rotation_errors_teaser = {p: [] for p in outlier_rates}
rotation_errors_ransac = {p: [] for p in outlier_rates}

# -------------------------
# 3) Run Monte Carlo Experiments
# -------------------------
for p in outlier_rates:
    outlier_ratio = p / 100.0
    for _ in range(num_trials):
        B, _, gt_R, _, _ = generate_transformed_cloud(
            A,
            outlier_ratio=outlier_ratio,
            max_s=1,
            max_t=0,
        )

        # --- TEASER++ Solver for Rotation Estimation ---
        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.noise_bound = 0.01
        solver_params.estimate_scaling = False
        solver_params.use_max_clique = True
        solver_params.rotation_estimation_algorithm = (
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        )
        solver_params.rotation_gnc_factor = 1.4
        solver_params.rotation_max_iterations = 100
        solver_params.rotation_cost_threshold = 1e-12

        solver = teaserpp_python.RobustRegistrationSolver(solver_params)
        solver.solve(A.T, B.T)
        solution = solver.getSolution()
        est_R_teaser = solution.rotation

        # Compute TEASER++ rotation error (in degrees)
        cos_angle = (np.trace(est_R_teaser.T @ gt_R) - 1) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure numerical stability
        rot_err_teaser = np.degrees(np.arccos(cos_angle))
        rotation_errors_teaser[p].append(rot_err_teaser)

        # --- RANSAC-Based Rotation Estimation ---
        def estimate_rotation_ransac(A, B, threshold=0.01, max_trials=1000):
            """Estimates the optimal rotation matrix using RANSAC."""
            num_points = A.shape[0]
            best_inliers = 0
            best_rotation = np.eye(3)

            for _ in range(max_trials):
                # Randomly sample 3 non-collinear points
                sample_indices = np.random.choice(num_points, 3, replace=False)
                A_sample = A[sample_indices]
                B_sample = B[sample_indices]

                # Compute rotation using SVD
                H = A_sample.T @ B_sample
                U, _, Vt = np.linalg.svd(H)
                R_est = Vt.T @ U.T

                # Ensure proper rotation (det(R) = 1)
                if np.linalg.det(R_est) < 0:
                    Vt[-1, :] *= -1
                    R_est = Vt.T @ U.T

                # Apply rotation to all points
                A_rotated = A @ R_est.T

                # Count inliers based on threshold
                distances = np.linalg.norm(A_rotated - B, axis=1)
                inliers = np.sum(distances < threshold)

                # Keep the best rotation matrix
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_rotation = R_est

            return best_rotation

        # Estimate rotation using RANSAC
        est_R_ransac = estimate_rotation_ransac(A, B)

        # Compute RANSAC rotation error (in degrees)
        cos_angle_ransac = (np.trace(est_R_ransac.T @ gt_R) - 1) / 2.0
        cos_angle_ransac = np.clip(cos_angle_ransac, -1.0, 1.0)  # Numerical safety
        rot_err_ransac = np.degrees(np.arccos(cos_angle_ransac))
        rotation_errors_ransac[p].append(rot_err_ransac)


# -------------------------
# 4) Group Errors into Bins for Plotting
# -------------------------
def group_errors(error_dict, group_size=1):
    groups = {}
    sorted_keys = sorted(error_dict.keys())
    for key in sorted_keys:
        group_idx = (key - 1) // group_size
        group_label = f"{group_idx*group_size+1}-{(group_idx+1)*group_size}"
        if group_label not in groups:
            groups[group_label] = []
        groups[group_label].extend(error_dict[key])
    return groups


rot_groups_teaser = group_errors(rotation_errors_teaser, group_size=10)
rot_groups_ransac = group_errors(rotation_errors_ransac, group_size=10)

# Sort group labels for x-axis
group_labels = sorted(rot_groups_teaser.keys(), key=lambda x: int(x.split("-")[0]))
box_data_teaser = [rot_groups_teaser[label] for label in group_labels]
box_data_ransac = [rot_groups_ransac[label] for label in group_labels]

# -------------------------
# 5) Plot TEASER++ vs RANSAC Rotation Errors
# -------------------------
fig, ax = plt.subplots(figsize=(10, 6))
positions = np.arange(len(group_labels))

# Plot TEASER++ boxplot
bp_teaser = ax.boxplot(
    box_data_teaser,
    positions=positions + 0.15,  # Shift for side-by-side comparison
    widths=0.25,
    patch_artist=True,
    showfliers=False,
    boxprops=dict(facecolor="lightblue", alpha=0.7),
    medianprops=dict(color="black"),
)

# Plot RANSAC boxplot
bp_ransac = ax.boxplot(
    box_data_ransac,
    positions=positions - 0.15,  # Shift for side-by-side comparison
    widths=0.25,
    patch_artist=True,
    showfliers=False,
    boxprops=dict(facecolor="red", alpha=0.7),
    medianprops=dict(color="black"),
)

# Formatting
ax.set_xticks(positions)
ax.set_xticklabels(group_labels)
ax.set_xlabel("Outlier Rate Group (%)")
ax.set_ylabel("Rotation Error (deg)")
ax.set_title("Rotation Estimation Error: TEASER++ vs RANSAC")
ax.set_yscale("log")
ax.set_ylim(1e-1, 100)
ax.grid(True, which="both", linestyle="--")

# Legend
handles = [
    plt.Rectangle((0, 0), 1, 1, color="lightblue", alpha=0.7, label="TEASER++"),
    plt.Rectangle((0, 0), 1, 1, color="red", alpha=0.7, label="RANSAC"),
]
ax.legend(handles=handles, loc="upper left")

plt.tight_layout()
plt.show()

print(f"Point cloud has {A.shape[0]} points.")
