import numpy as np
import matplotlib.pyplot as plt
import teaserpp_python
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from sklearn.linear_model import RANSACRegressor
from data import generate_transformed_cloud

# -------------------------
# 1) Load and normalize Bunny point cloud
# -------------------------
PLY_FILE = "bunny.ply"
bunny_pcd = o3d.io.read_point_cloud(PLY_FILE)
# Voxel downsampling (adjust voxel size as needed)
downsampled_pcd = bunny_pcd.voxel_down_sample(voxel_size=0.04)
A = np.asarray(downsampled_pcd.points)

# Normalize to fit inside [0,1]^3
A = (A - np.min(A, axis=0)) / (np.max(A, axis=0) - np.min(A, axis=0))


# -------------------------
# 2) Function to generate random transformations + outliers
# -------------------------


# -------------------------
# 3) Monte Carlo Simulation Parameters
# -------------------------
num_trials = 40
outlier_rates = np.arange(0, 95, 10)  # 0, 10, 20, ..., 90

# We'll store a *list of lists* so that scale_errors_tls[i] is
# the list of scale errors for TEASER++ at outlier_rates[i],
# similarly for scale_errors_ransac, rotation_errors, translation_errors.
scale_errors_tls = []
scale_errors_ransac = []
rotation_errors = []
translation_errors = []

# -------------------------
# 4) Run Monte Carlo simulations
# -------------------------
for outlier_ratio in outlier_rates:
    scale_err_tls_thisOR = []
    scale_err_ransac_thisOR = []
    rot_err_thisOR = []
    trans_err_thisOR = []

    for _ in range(num_trials):
        B, gt_s, gt_R, gt_t, _ = generate_transformed_cloud(
            A, outlier_ratio=outlier_ratio / 100.0
        )

        # --- TEASER++ Solver (TLS) ---
        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.noise_bound = 0.01
        solver_params.estimate_scaling = True
        solver_params.rotation_estimation_algorithm = (
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        )
        solver_params.rotation_gnc_factor = 1.4
        solver_params.rotation_max_iterations = 100
        solver_params.rotation_cost_threshold = 1e-12

        solver = teaserpp_python.RobustRegistrationSolver(solver_params)
        solver.solve(A.T, B.T)
        solution = solver.getSolution()

        est_s_tls = solution.scale
        est_R = solution.rotation
        est_t = solution.translation

        # --- RANSAC scale estimate (using norms) ---
        # Compute centroids of A and B
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)

        # Compute distances from each point to the centroid
        distances_A = np.linalg.norm(A - centroid_A, axis=1).reshape(-1, 1)
        distances_B = np.linalg.norm(B - centroid_B, axis=1)

        ransac = RANSACRegressor()
        ransac.fit(distances_A, distances_B)
        est_s_ransac = ransac.estimator_.coef_[0]

        # --- Compute errors ---
        scale_err_tls_thisOR.append(np.abs(est_s_tls - gt_s))
        scale_err_ransac_thisOR.append(np.abs(est_s_ransac - gt_s))

        # Convert rotation error to degrees:
        #   error = arccos( (trace(R_est^T R_gt) - 1)/2 ), in radians -> convert to deg
        cos_angle = (np.trace(est_R.T @ gt_R) - 1) / 2
        # Numerical issues can push cos_angle slightly out of [-1,1], so clip it:
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        rot_err_degs = np.degrees(np.arccos(cos_angle))
        rot_err_thisOR.append(rot_err_degs)

        trans_err_thisOR.append(np.linalg.norm(est_t - gt_t))

    # Store lists of errors for this outlier ratio
    scale_errors_tls.append(scale_err_tls_thisOR)
    scale_errors_ransac.append(scale_err_ransac_thisOR)
    rotation_errors.append(rot_err_thisOR)
    translation_errors.append(trans_err_thisOR)

# -------------------------
# 5) Plot Results
# -------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# -------------------------
# (a) Scale Estimation vs. Outlier Rate
#     Grouped boxplots: TEASER++ (TLS) and RANSAC side by side
# -------------------------
# We'll shift TLS boxes slightly left, RANSAC slightly right
shift = 2.0  # how far we shift the positions
positions_tls = outlier_rates + shift
positions_ransac = outlier_rates - shift

# TLS boxplots
bp_tls = axs[0, 0].boxplot(
    scale_errors_tls,
    positions=positions_tls,
    widths=3.5,
    patch_artist=True,
    labels=[None] * len(outlier_rates),  # We'll manually set x-ticks
)
# Give TLS boxes a color
for box in bp_tls["boxes"]:
    box.set(facecolor="lightblue")

# RANSAC boxplots
bp_ransac = axs[0, 0].boxplot(
    scale_errors_ransac,
    positions=positions_ransac,
    widths=3.5,
    patch_artist=True,
    labels=[None] * len(outlier_rates),
)
# Give RANSAC boxes a different color
for box in bp_ransac["boxes"]:
    box.set(facecolor="red")

axs[0, 0].set_yscale("log")
axs[0, 0].set_xticks(outlier_rates)
axs[0, 0].set_xlabel("Outlier Rate (%)")
axs[0, 0].set_ylabel("Scale Error")
axs[0, 0].set_title("(a) Scale Estimation")
axs[0, 0].legend(
    [bp_tls["boxes"][0], bp_ransac["boxes"][0]], ["TLS", "RANSAC"], loc="upper left"
)

# -------------------------
# (b) [Optional] Empty or Additional Plot
#     If you have a second method, you can place it here
#     For now, let's leave it blank or re-use it as needed.
# -------------------------
axs[0, 1].axis("off")
axs[0, 1].set_title("Placeholder or Additional Plot")

# -------------------------
# (c) Rotation Estimation vs. Outlier Rate
# -------------------------
axs[1, 0].boxplot(rotation_errors, positions=outlier_rates, widths=5)
axs[1, 0].set_yscale("log")
axs[1, 0].set_xlabel("Outlier Rate (%)")
axs[1, 0].set_ylabel("Rotation Error [deg]")
axs[1, 0].set_title("(c) Rotation Estimation")
axs[1, 0].set_xticks(outlier_rates)

# -------------------------
# (d) Translation Estimation vs. Outlier Rate
# -------------------------
axs[1, 1].boxplot(translation_errors, positions=outlier_rates, widths=5)
axs[1, 1].set_yscale("log")
axs[1, 1].set_xlabel("Outlier Rate (%)")
axs[1, 1].set_ylabel("Translation Error [m]")
axs[1, 1].set_title("(d) Translation Estimation")
axs[1, 1].set_xticks(outlier_rates)

plt.tight_layout()
plt.show()
