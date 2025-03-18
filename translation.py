import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import teaserpp_python  # Make sure TEASER++ is installed and accessible
from data import load_bunny, generate_transformed_cloud

# -------------------------
# 1) Load and downsample bunny, then normalize to [0,1]^3
# -------------------------
A = load_bunny(N=400)


# -------------------------
# 2) RANSAC for translation only (ignoring rotation)
#     We assume the model: B[i] = A[i] + t
# -------------------------
def ransac_translation_only(A, B, distance_threshold=0.02, max_iter=1000):

    N = A.shape[0]
    best_inliers = 0
    t_best = np.zeros(3)
    for _ in range(max_iter):
        i = np.random.randint(0, N)  # pick one random correspondence
        t_est = B[i] - A[i]
        A_transformed = A + t_est
        residuals = np.linalg.norm(A_transformed - B, axis=1)
        inliers = np.sum(residuals < distance_threshold)
        if inliers > best_inliers:
            best_inliers = inliers
            t_best = t_est
    return t_best


# -------------------------
# 3) Main Loop: MC Estimates for TEASER++ vs. RANSAC
# -------------------------
num_runs_per_rate = 10
# Outlier percentages from 1 to 99%
outlier_rates = np.arange(1, 100)

# Dictionaries to store translation errors for each outlier percentage
teaser_trans_errors = {p: [] for p in outlier_rates}
ransac_trans_errors = {p: [] for p in outlier_rates}

for p in outlier_rates:
    outlier_ratio = p / 100.0
    for _ in range(num_runs_per_rate):
        # Generate transformed cloud B with outliers
        B, _, _, t_gt, _ = generate_transformed_cloud(
            A, outlier_ratio=outlier_ratio, Rot=False, max_s=1, max_t=3
        )

        # ----- TEASER++: Solve for full transformation (scale fixed to 1) -----
        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.noise_bound = 0.01
        solver_params.estimate_scaling = False  # scale is fixed to 1
        solver_params.rotation_estimation_algorithm = (
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        )
        solver_params.rotation_gnc_factor = 1.4
        solver_params.rotation_max_iterations = 100
        solver_params.rotation_cost_threshold = 1e-12
        solver_params.use_max_clique = False
        solver_params.inlier_selection_mode = (
            teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
        )
        solver = teaserpp_python.RobustRegistrationSolver(solver_params)
        solver.solve(A.T, B.T)
        solution = solver.getSolution()
        t_est_teaser = solution.translation  # TEASER++ estimated translation
        print(f"ground thruth translation is {t_gt}")
        print(f"estimated translation is {t_est_teaser}")
        teaser_error = np.linalg.norm(t_est_teaser - t_gt)
        teaser_trans_errors[p].append(teaser_error)

        # ----- RANSAC: Solve for translation only -----
        t_est_ransac = ransac_translation_only(
            A, B, distance_threshold=0.02, max_iter=1000
        )
        ransac_error = np.linalg.norm(t_est_ransac - t_gt)
        ransac_trans_errors[p].append(ransac_error)


# -------------------------
# 4) Group errors by dozen: 1-10, 11-20, ..., 91-99
# -------------------------
def group_errors(error_dict, group_size=10):
    groups = {}
    # Sort keys to be safe
    sorted_keys = sorted(error_dict.keys())
    for key in sorted_keys:
        # Determine group index: 1-10 -> group 0, 11-20 -> group 1, etc.
        group_idx = (key - 1) // group_size
        group_label = f"{group_idx*group_size+1}-{(group_idx+1)*group_size}"
        if group_label not in groups:
            groups[group_label] = []
        groups[group_label].extend(error_dict[key])
    return groups


teaser_groups = group_errors(teaser_trans_errors, group_size=1)
ransac_groups = group_errors(ransac_trans_errors, group_size=1)

# Sort groups by numeric order based on their lower bound
group_labels = sorted(teaser_groups.keys(), key=lambda x: int(x.split("-")[0]))
teaser_box_data = [teaser_groups[label] for label in group_labels]
ransac_box_data = [ransac_groups[label] for label in group_labels]

# -------------------------
# 5) Plot comparison of translation errors vs. grouped outlier ratio
# -------------------------
fig, ax = plt.subplots(figsize=(10, 6))
positions = np.arange(len(group_labels))
width = 0.3

bp_teaser = ax.boxplot(
    teaser_box_data,
    positions=positions + width / 2,
    widths=0.25,
    patch_artist=True,
    showfliers=False,
)
bp_ransac = ax.boxplot(
    ransac_box_data,
    positions=positions - width / 2,
    widths=0.25,
    patch_artist=True,
    showfliers=False,
)

for box in bp_teaser["boxes"]:
    box.set(facecolor="lightblue", alpha=0.7)
for box in bp_ransac["boxes"]:
    box.set(facecolor="red", alpha=0.7)

ax.set_xticks(positions)
ax.set_xticklabels(group_labels)
ax.set_xlabel("Outlier Ratio Group (%)")
ax.set_ylabel("Translation Error (L2 norm)")
ax.set_title("Translation Error: TEASER++ vs. RANSAC (Grouped by Outlier Ratio)")
ax.set_yscale("log")
ax.set_ylim(1e-3, 10)
ax.grid(True, which="both", ls="--")

blue_patch = plt.Line2D(
    [], [], color="lightblue", marker="s", linestyle="", markersize=10, label="TEASER++"
)
green_patch = plt.Line2D(
    [], [], color="red", marker="s", linestyle="", markersize=10, label="RANSAC"
)
ax.legend(handles=[blue_patch, green_patch])

plt.tight_layout()
plt.show()
