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
A = load_bunny(N=1000)
# -------------------------
# 2) Monte Carlo Simulation Parameters
# -------------------------
num_trials = 10
outlier_rates = np.arange(1, 100)

# Dictionaries to store scale errors for each exact outlier rate
scale_errors_tls = {p: [] for p in outlier_rates}
scale_errors_ransac = {p: [] for p in outlier_rates}

for p in outlier_rates:
    outlier_ratio = p / 100.0
    for _ in range(num_trials):
        B, gt_s, _, _, _ = generate_transformed_cloud(
            A, outlier_ratio=outlier_ratio, max_s=2
        )
        print(f"ground thruth scale is {gt_s}")

        # --- TEASER++ Solver (TLS) for Scale Estimation ---
        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.cbar2 = 1
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
        print(f"estimated TLS scale is {est_s_tls}")

        # --- RANSAC Scale Estimate (using point-to-centroid distances) ---
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        distances_A = np.linalg.norm(A - centroid_A, axis=1).reshape(-1, 1)
        distances_B = np.linalg.norm(B - centroid_B, axis=1)
        ransac = RANSACRegressor()
        ransac.fit(distances_A, distances_B)
        est_s_ransac = ransac.estimator_.coef_[0]

        # Record absolute scale error
        scale_errors_tls[p].append(np.abs(est_s_tls - gt_s))
        scale_errors_ransac[p].append(np.abs(est_s_ransac - gt_s))


# -------------------------
# 3) Group errors into bins (e.g., 1-10%, 11-20%, …, 91-99%)
# -------------------------
def group_errors(error_dict, group_size=10):
    groups = {}
    sorted_keys = sorted(error_dict.keys())
    for key in sorted_keys:
        group_idx = (key - 1) // group_size
        group_label = f"{group_idx*group_size+1}-{(group_idx+1)*group_size}"
        if group_label not in groups:
            groups[group_label] = []
        groups[group_label].extend(error_dict[key])
    return groups


tls_groups = group_errors(scale_errors_tls, group_size=10)
ransac_groups = group_errors(scale_errors_ransac, group_size=10)

# Sort group labels by their numeric lower bound
group_labels = sorted(tls_groups.keys(), key=lambda x: int(x.split("-")[0]))
tls_box_data = [tls_groups[label] for label in group_labels]
ransac_box_data = [ransac_groups[label] for label in group_labels]

# -------------------------
# 4) Plot Scale Estimation Errors: TEASER++ (TLS) vs. RANSAC
# -------------------------
fig, ax = plt.subplots(figsize=(10, 6))
positions = np.arange(len(group_labels))
width = 0.3

bp_tls = ax.boxplot(
    tls_box_data,
    positions=positions - width / 2,
    widths=0.25,
    patch_artist=True,
    showfliers=False,
)
bp_ransac = ax.boxplot(
    ransac_box_data,
    positions=positions + width / 2,
    widths=0.25,
    patch_artist=True,
    showfliers=False,
)

# Set box colors
for box in bp_tls["boxes"]:
    box.set(facecolor="lightblue", alpha=0.7)
for box in bp_ransac["boxes"]:
    box.set(facecolor="red", alpha=0.7)

ax.set_xticks(positions)
ax.set_xticklabels(group_labels)
ax.set_xlabel("Outlier Rate Group (%)")
ax.set_ylabel("Scale Estimation Error")
ax.set_title("Scale Estimation: TEASER++ (TLS) vs. RANSAC")
ax.set_yscale("log")
ax.set_ylim(1e-4, 100)
ax.grid(True, which="both", ls="--")
blue_patch = plt.Line2D(
    [], [], color="lightblue", marker="s", linestyle="", markersize=10, label="TLS"
)
red_patch = plt.Line2D(
    [], [], color="red", marker="s", linestyle="", markersize=10, label="RANSAC"
)
ax.legend(handles=[blue_patch, red_patch])
plt.tight_layout()
plt.show()

print(f"Point cloud has {A.shape[0]} points.")
