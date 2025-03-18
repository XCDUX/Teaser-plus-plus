import numpy as np
import matplotlib.pyplot as plt
import teaserpp_python
from data import generate_transformed_cloud, load_bunny

# -------------------------
# 1) Load and downsample bunny, then normalize to [0,1]^3
# -------------------------
A = load_bunny(N=1000)

# -------------------------
# 2) Simulation parameters for MCIS plot
# -------------------------
num_trials = 10
outlier_percentages = [50, 60, 70, 80, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
# outlier_percentages = [91, 92, 93, 94, 95, 96, 97, 98, 99]

after_mcis = {p: [] for p in outlier_percentages}

# -------------------------
# 3) Monte Carlo simulation
# -------------------------

for p in outlier_percentages:
    outlier_ratio = p / 100.0
    for _ in range(num_trials):
        # Generate transformed cloud with fixed scale and injected outliers
        B, _, _, _, out_mask = generate_transformed_cloud(
            A, outlier_ratio=outlier_ratio, max_s=2, max_t=1, noise_std=0.01
        )

        # "Before MCIS": simply the ratio of points replaced as outliers.
        before_ratio = np.sum(out_mask) / A.shape[0] * 100.0

        # Run TEASER++ with known scale (set estimate_scaling to False)
        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.cbar2 = 1
        solver_params.noise_bound = 0.01
        solver_params.estimate_scaling = False  # scale is fixed to 1
        solver_params.rotation_estimation_algorithm = (
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        )
        solver_params.rotation_gnc_factor = 1.4
        solver_params.rotation_max_iterations = 100
        solver_params.rotation_cost_threshold = 1e-12

        solver_params.use_max_clique = True
        solver = teaserpp_python.RobustRegistrationSolver(solver_params)
        solver.solve(A.T, B.T)

        # Get the indices of the maximum clique (inlier set after MCIS)
        clique_indices = solver.getInlierMaxClique()

        if len(clique_indices) == 0:
            # If no clique is found, treat as 100% outliers
            after_ratio = 100.0
        else:
            # Count how many of the points in the clique are outliers
            num_outliers_in_clique = np.sum(out_mask[clique_indices])
            after_ratio = num_outliers_in_clique / float(len(clique_indices)) * 100.0
            print(after_ratio)
        after_mcis[p].append(after_ratio)

# -------------------------
fig, ax = plt.subplots(figsize=(10, 6))
positions = np.arange(len(outlier_percentages))
# Plot the identity (dotted line); this represents "Before MCIS"
ax.plot(positions, outlier_percentages, "k--", label="Before MCIS")

# Prepare boxplot data for "After MCIS": list of lists for each x position.
box_data = [after_mcis[p] for p in outlier_percentages]
bp = ax.boxplot(box_data, positions=positions, widths=0.25, patch_artist=True)
for box in bp["boxes"]:
    box.set(facecolor="lightblue", alpha=0.7)

ax.set_xlabel("Outlier Ratio (%)")
ax.set_ylabel("MCIS Outlier Ratio (%)")
ax.set_title("MCIS: Outlier Ratio Before vs. After MCIS")
ax.set_xticks(positions)
ax.set_xticklabels(outlier_percentages)  # Set actual values on x-axis
ax.legend()

plt.tight_layout()
plt.show()

print(
    f"Point cloud has {A.shape[0]} points."
)  # for the sake of checking at the end of compilation that we worked on the right cloud.
