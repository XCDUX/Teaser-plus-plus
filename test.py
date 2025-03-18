import os
import numpy as np
import teaserpp_python
from data import generate_rand_clouds
import time


def compute_outlier_percentage(inlier_mask):
    """
    Given a boolean numpy array (mask) indicating inliers (True) and outliers (False),
    compute the percentage of rejected (outlier) measurements, along with inlier count and total.
    """
    total = inlier_mask.size
    inlier_count = np.count_nonzero(inlier_mask)
    rejected = total - inlier_count
    return (rejected / total) * 100, inlier_count, total


def save_array_csv(array, filename):
    """
    Save a numpy array to a CSV file.
    """
    np.savetxt(filename, array, delimiter=",", fmt="%g")
    print(f"Saved array with shape {array.shape} to {filename}")


# Create CSV folder if it doesn't exist.
os.makedirs("CSV", exist_ok=True)

# Generate random clouds (source, destination, and ground truth transformation)
src, dst, gt_scale, gt_translation, gt_rotation = generate_rand_clouds(
    n_points=5000, n_random_outliers=4990, std_noise=0
)

# Set up TEASER++ solver parameters
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

# Create the solver and solve the registration problem
solver = teaserpp_python.RobustRegistrationSolver(solver_params)
start_time = time.time()
solver.solve(src, dst)
end_time = time.time()
elapsed_time = end_time - start_time

solution = solver.getSolution()

# Print out the estimated transformation parameters along with the ground truth.
print("Transformation Comparison:")
print("--------------------------------------------------")
print("Scale:")
print("  Estimated: {:.4f}".format(solution.scale))
print("  Ground Truth: {:.4f}".format(gt_scale))
print("\nTranslation:")
print("  Estimated:\n", solution.translation)
print("  Ground Truth:\n", gt_translation)
print("\nRotation:")
print("  Estimated:\n", solution.rotation)
print("  Ground Truth:\n", gt_rotation)

# Retrieve inlier masks from TEASER++.
scale_inliers_mask = solver.getScaleInliersMask()  # Boolean mask for scale step
max_clique_inliers = (
    solver.getInlierMaxClique()
)  # Either boolean mask or inlier indices for max clique
rotation_inliers_mask = (
    solver.getRotationInliersMask()
)  # Boolean mask for rotation step
translation_inliers_mask = (
    solver.getTranslationInliersMask()
)  # Boolean mask for translation step

# Retrieve the corresponding pairing maps.
scale_map = solver.getScaleInliersMap()  # shape: (2, num_scale_measurements)
rotation_map = solver.getRotationInliersMap()  # shape: (1, num_rotation_measurements)
translation_map = (
    solver.getTranslationInliersMap()
)  # shape: (1, num_translation_measurements)
src_tim_map = solver.getSrcTIMsMap()  # shape: (2, num_measurements) for max clique

# Compute outlier percentages and counts.
scale_outlier_percentage, scale_inlier_count, total_scale = compute_outlier_percentage(
    scale_inliers_mask
)
rotation_outlier_percentage, rotation_inlier_count, total_rotation = (
    compute_outlier_percentage(rotation_inliers_mask)
)
translation_outlier_percentage, translation_inlier_count, total_translation = (
    compute_outlier_percentage(translation_inliers_mask)
)

# For max clique, handle both boolean mask and index list.
if isinstance(max_clique_inliers, np.ndarray) and max_clique_inliers.dtype == np.bool_:
    max_clique_outlier_percentage, max_clique_inlier_count, total_max_clique = (
        compute_outlier_percentage(max_clique_inliers)
    )
    max_clique_outlier_indices = np.where(~max_clique_inliers)[0]
else:
    max_clique_inlier_count = len(max_clique_inliers)
    total_max_clique = scale_inliers_mask.size  # assuming same total count
    all_indices = set(range(total_max_clique))
    max_clique_outlier_indices = np.array(
        sorted(all_indices.difference(set(max_clique_inliers)))
    )
    rejected = total_max_clique - max_clique_inlier_count
    max_clique_outlier_percentage = (rejected / total_max_clique) * 100

# Instead of saving indices, select the pairing values from the maps.
# For scale (2-column pairing):
scale_outlier_indices = np.where(~scale_inliers_mask)[0]
scale_pairings_outliers = scale_map[
    :, scale_outlier_indices
]  # shape: (2, num_outliers)

# For rotation (1-column pairing), we assume the outlier mask directly indexes original correspondences.
rotation_outlier_indices = np.where(~rotation_inliers_mask)[0]
rotation_outliers = rotation_map[rotation_outlier_indices]  # 1D array

# For translation (1-column pairing):
translation_outlier_indices = np.where(~translation_inliers_mask)[0]
translation_outliers = translation_map[translation_outlier_indices]  # 1D array

# For max clique, use the source TIM map as the pairing information.
max_clique_pairings_inliers = src_tim_map[
    :, max_clique_inliers
]  # shape: (2, num_inliers)

# Save max clique inliers to CSV
save_array_csv(
    max_clique_pairings_inliers.T,
    os.path.join("CSV", "max_clique_pairings_inliers.csv"),
)
# Save each outlier pairing array to CSV in the CSV folder.
""" save_array_csv(
    scale_pairings_outliers.T, os.path.join("CSV", "scale_pairings_outliers.csv")
) """

save_array_csv(
    rotation_outliers.reshape(-1, 1), os.path.join("CSV", "rotation_outliers.csv")
)
save_array_csv(
    translation_outliers.reshape(-1, 1), os.path.join("CSV", "translation_outliers.csv")
)

save_array_csv(src.T, os.path.join("CSV", "source_cloud.csv"))

# Save estimated transformation parameters.
save_array_csv(solution.rotation, os.path.join("CSV", "estimated_rotation.csv"))
save_array_csv(solution.translation, os.path.join("CSV", "estimated_translation.csv"))
# Save scale as a one-row CSV.
np.savetxt(
    os.path.join("CSV", "estimated_scale.csv"),
    [solution.scale],
    delimiter=",",
    fmt="%f",
)
print(f"Saved estimated scale to {os.path.join('CSV','estimated_scale.csv')}")

# Print final statistics.
print("\n--- TEASER++ Registration Statistics ---")
print("Registration Time: {:.4f} seconds".format(elapsed_time))
print(
    "Scale estimation: {:.2f}% outliers rejected ({} inliers out of {} measurements)".format(
        scale_outlier_percentage, scale_inlier_count, total_scale
    )
)
print(
    "Max Clique pruning: {:.2f}% outliers rejected ({} inliers out of {} measurements)".format(
        max_clique_outlier_percentage, max_clique_inlier_count, total_max_clique
    )
)
print(
    "Rotation estimation: {:.2f}% outliers rejected ({} inliers out of {} measurements)".format(
        rotation_outlier_percentage, rotation_inlier_count, total_rotation
    )
)
print(
    "Translation estimation: {:.2f}% outliers rejected ({} inliers out of {} measurements)".format(
        translation_outlier_percentage, translation_inlier_count, total_translation
    )
)
