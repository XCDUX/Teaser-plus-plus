import os
import numpy as np
import teaserpp_python
from data import generate_rand_clouds
import time
import shutil
import pandas as pd


def compute_outlier_percentage(inlier_mask):
    total = inlier_mask.size
    inlier_count = np.count_nonzero(inlier_mask)
    rejected = total - inlier_count
    return (rejected / total) * 100, inlier_count, total


def save_array_csv(array, filename):
    np.savetxt(filename, array, delimiter=",", fmt="%g")
    print(f"Saved array with shape {array.shape} to {filename}")


def should_save_rotation(estimated_rotation, ground_truth_rotation, threshold=0.2):
    return (
        np.linalg.norm(estimated_rotation - ground_truth_rotation, ord="fro")
        > threshold
    )


def archive_results(iteration):
    archive_folder = os.path.join("CSV", f"CSV_{iteration}")
    os.makedirs(archive_folder, exist_ok=True)

    for file in os.listdir("CSV"):
        file_path = os.path.join("CSV", file)
        if os.path.isfile(file_path) and not file.endswith(".zip"):
            shutil.move(file_path, os.path.join(archive_folder, file))

    archive_path = os.path.join("CSV", f"CSV_{iteration}.zip")
    shutil.make_archive(archive_path.replace(".zip", ""), "zip", archive_folder)
    shutil.rmtree(archive_folder)
    print(f"Archived results to {archive_path}")


os.makedirs("CSV", exist_ok=True)
iteration = 0
max_iterations = 100  # Adjust as needed

while iteration < max_iterations:
    print(f"Iteration {iteration}...")

    src, dst, gt_scale, gt_translation, gt_rotation = generate_rand_clouds(
        n_points=1000,
        n_random_outliers=990,
        std_noise=0,
        max_scale=10,
        max_translation=10,
        save_path="CSV",
    )

    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = 1e-1
    solver_params.estimate_scaling = True
    solver_params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    start_time = time.time()
    solver.solve(src, dst)
    elapsed_time = time.time() - start_time

    solution = solver.getSolution()

    max_clique_inliers = solver.getInlierMaxClique()
    max_clique_pairings_inliers = solver.getSrcTIMsMap()[:, max_clique_inliers]

    scale_inliers = np.array(solver.getScaleInliers())

    print("Estimated vs Ground Truth Rotation Difference:")
    print(np.linalg.norm(solution.rotation - gt_rotation, ord="fro"))

    if should_save_rotation(solution.rotation, gt_rotation):
        save_array_csv(src.T, os.path.join("CSV", "source_cloud.csv"))
        save_array_csv(solution.rotation, os.path.join("CSV", "estimated_rotation.csv"))
        save_array_csv(
            solution.translation, os.path.join("CSV", "estimated_translation.csv")
        )
        np.savetxt(
            os.path.join("CSV", "estimated_scale.csv"),
            [solution.scale],
            delimiter=",",
            fmt="%f",
        )

        save_array_csv(gt_rotation, os.path.join("CSV", "ground_truth_rotation.csv"))
        save_array_csv(
            gt_translation, os.path.join("CSV", "ground_truth_translation.csv")
        )
        np.savetxt(
            os.path.join("CSV", "ground_truth_scale.csv"),
            [gt_scale],
            delimiter=",",
            fmt="%f",
        )

        save_array_csv(
            max_clique_pairings_inliers.T,
            os.path.join("CSV", "max_clique_pairings_inliers.csv"),
        )

        save_array_csv(
            scale_inliers,
            os.path.join("CSV", "scale_inliers_pairings.csv"),
        )

        archive_results(iteration)

        iteration += 1
