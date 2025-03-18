import numpy as np
import open3d as o3d
import teaserpp_python
import tempfile
import os

# Hypothetical Python wrapper for Go-ICP
from py_goicp import GoICP, POINT3D, ROTNODE, TRANSNODE
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from data import load_bunny, generate_transformed_cloud


def translation_error(t_est, t_gt):
    return np.linalg.norm(t_est - t_gt)


def rotation_error_deg(R_est, R_gt):
    # R_est, R_gt are 3x3 rotation matrices
    cos_angle = (np.trace(R_est.T @ R_gt) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def generate_partial_overlap_clouds(A, overlap_percentage=80):
    # 1) Generate a random rotation and translation
    R_gt = R.random().as_matrix()
    t_gt = np.random.uniform(-15, 15, size=3)

    # 2) Create a fully overlapped B by applying R_gt and t_gt to A
    B_full = A @ R_gt.T + t_gt

    # 3) Remove a fraction of points from B_full
    N = B_full.shape[0]
    overlap_fraction = overlap_percentage / 100.0
    num_remove = int((1 - overlap_fraction) * N)
    print(f"points removed:{num_remove}")
    print(R_gt)
    print(t_gt)
    # Randomly choose 'num_remove' indices to discard
    remove_indices = np.random.choice(N, num_remove, replace=False)
    mask = np.ones(N, dtype=bool)
    mask[remove_indices] = False
    B = B_full[mask]

    return A, B, R_gt, t_gt


def generate_correspondences(A, B):
    N, M = A.shape[0], B.shape[0]

    # Repeat each point in A for every point in B
    A_cor = np.repeat(A, M, axis=0)  # Shape (N*M, 3)

    # Tile B so it matches the repeated A points
    B_cor = np.tile(B, (N, 1))  # Shape (N*M, 3)

    return A_cor, B_cor


def teaser_registration(A, B):
    # A, B are Nx3 arrays
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

    R_est = solution.rotation
    t_est = solution.translation

    print(R_est)
    return R_est, t_est


def write_temp_pointcloud(arr):
    # arr is an (N,3) array of points
    tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt")
    tmp.write("# x y z\n")
    np.savetxt(tmp, arr, fmt="%f")
    tmp.close()
    return tmp.name


def goicp_registration(A, B):
    # Write A and B to temporary files in the expected format.
    model_file = write_temp_pointcloud(A)
    data_file = write_temp_pointcloud(B)

    # Use the provided loadPointCloud function from the GoICP example.

    def loadPointCloud(filename):
        pcloud = np.loadtxt(filename, skiprows=1)
        plist = pcloud.tolist()
        p3dlist = []
        for x, y, z in plist:
            pt = POINT3D(x, y, z)
            p3dlist.append(pt)
        return pcloud.shape[0], p3dlist, pcloud

    a_points = [POINT3D(0.0, 0.0, 0.0), POINT3D(0.5, 1.0, 0.0), POINT3D(1.0, 0.0, 0.0)]
    b_points = [POINT3D(0.0, 0.0, 0.0), POINT3D(1.0, 0.5, 0.0), POINT3D(1.0, -0.5, 0.0)]

    Nm, a_points, np_a_points = loadPointCloud(model_file)
    Nd, b_points, np_b_points = loadPointCloud(data_file)

    goicp = GoICP()
    rNode = ROTNODE()
    tNode = TRANSNODE()

    rNode.a = -3.1416
    rNode.b = -3.1416
    rNode.c = -3.1416
    rNode.w = 6.2832

    tNode.x = -0.5
    tNode.y = -0.5
    tNode.z = -0.5
    tNode.w = 1.0
    goicp.MSEThresh = 0.003

    goicp.loadModelAndData(Nm, a_points, Nd, b_points)
    goicp.setDTSizeAndFactor(300, 2.0)
    goicp.setInitNodeRot(rNode)
    goicp.setInitNodeTrans(tNode)
    goicp.trimFraction = 0.6
    goicp.BuildDT()
    goicp.Register()

    R_est = np.array(goicp.optimalRotation())
    t_est = np.array(goicp.optimalTranslation())
    print(R_est)

    # Remove temporary files
    os.remove(model_file)
    os.remove(data_file)
    return R_est, t_est


def icp_registration(A, B):
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(A)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(B)

    threshold = 0.05
    init_transform = np.eye(4)

    reg_result = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        threshold,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    # Extract R, t
    transform = reg_result.transformation  # 4x4
    R_est = transform[:3, :3]
    t_est = transform[:3, 3]
    return R_est, t_est


def rotation_experiment(base_cloud, overlaps, n_runs=10):
    errors_teaser_rot = {ov: [] for ov in overlaps}
    errors_goicp_rot = {ov: [] for ov in overlaps}
    errors_icp_rot = {ov: [] for ov in overlaps}

    errors_teaser_trans = {ov: [] for ov in overlaps}
    errors_goicp_trans = {ov: [] for ov in overlaps}
    errors_icp_trans = {ov: [] for ov in overlaps}

    for ov in overlaps:
        for _ in range(n_runs):
            A, B, R_gt, t_gt = generate_partial_overlap_clouds(base_cloud, ov)
            A_cor, B_cor = generate_correspondences(A, B)

            # TEASER++
            R_est_t, t_est_t = teaser_registration(A_cor, B_cor)
            err_rot_t = rotation_error_deg(R_est_t, R_gt)
            err_trans_t = translation_error(t_est_t, t_gt)
            errors_teaser_rot[ov].append(err_rot_t)
            errors_teaser_trans[ov].append(err_trans_t)

            # Go-ICP
            R_est_g, t_est_g = goicp_registration(A, B)
            err_rot_g = rotation_error_deg(R_est_g, R_gt)
            err_trans_g = translation_error(t_est_g, t_gt)
            errors_goicp_rot[ov].append(err_rot_g)
            errors_goicp_trans[ov].append(err_trans_g)

            # ICP
            R_est_i, t_est_i = icp_registration(A, B)
            err_rot_i = rotation_error_deg(R_est_i, R_gt)
            err_trans_i = translation_error(t_est_i, t_gt)
            errors_icp_rot[ov].append(err_rot_i)
            errors_icp_trans[ov].append(err_trans_i)

    return (
        errors_teaser_rot,
        errors_goicp_rot,
        errors_icp_rot,
        errors_teaser_trans,
        errors_goicp_trans,
        errors_icp_trans,
    )


def plot_errors(
    errors_teaser_rot,
    errors_goicp_rot,
    errors_icp_rot,
    errors_teaser_trans,
    errors_goicp_trans,
    errors_icp_trans,
):
    overlaps = sorted(errors_teaser_rot.keys())
    positions = np.arange(len(overlaps))
    width = 0.2

    # Convert dict-of-lists to list-of-lists
    teaser_rot_data = [errors_teaser_rot[ov] for ov in overlaps]
    goicp_rot_data = [errors_goicp_rot[ov] for ov in overlaps]
    icp_rot_data = [errors_icp_rot[ov] for ov in overlaps]

    teaser_trans_data = [errors_teaser_trans[ov] for ov in overlaps]
    goicp_trans_data = [errors_goicp_trans[ov] for ov in overlaps]
    icp_trans_data = [errors_icp_trans[ov] for ov in overlaps]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # ---------------- Rotation Error Plot ----------------
    ax = axes[0]
    bp_teaser = ax.boxplot(
        teaser_rot_data,
        positions=positions - width,
        widths=0.15,
        patch_artist=True,
        showfliers=False,
        labels=[None] * len(overlaps),
    )
    bp_goicp = ax.boxplot(
        goicp_rot_data,
        positions=positions,
        widths=0.15,
        patch_artist=True,
        showfliers=False,
        labels=[None] * len(overlaps),
    )
    bp_icp = ax.boxplot(
        icp_rot_data,
        positions=positions + width,
        widths=0.15,
        patch_artist=True,
        showfliers=False,
        labels=[None] * len(overlaps),
    )

    for box in bp_teaser["boxes"]:
        box.set(facecolor="lightblue", alpha=0.7)
    for box in bp_goicp["boxes"]:
        box.set(facecolor="lightgreen", alpha=0.7)
    for box in bp_icp["boxes"]:
        box.set(facecolor="salmon", alpha=0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(overlaps)
    ax.set_xlabel("Overlap (%)")
    ax.set_ylabel("Rotation Error (deg)")
    ax.set_yscale("log")
    ax.set_title("Rotation Error vs. Overlap")
    ax.grid(True, which="both", ls="--")

    # ---------------- Translation Error Plot ----------------
    ax = axes[1]
    bp_teaser = ax.boxplot(
        teaser_trans_data,
        positions=positions - width,
        widths=0.15,
        patch_artist=True,
        showfliers=False,
        labels=[None] * len(overlaps),
    )
    bp_goicp = ax.boxplot(
        goicp_trans_data,
        positions=positions,
        widths=0.15,
        patch_artist=True,
        showfliers=False,
        labels=[None] * len(overlaps),
    )
    bp_icp = ax.boxplot(
        icp_trans_data,
        positions=positions + width,
        widths=0.15,
        patch_artist=True,
        showfliers=False,
        labels=[None] * len(overlaps),
    )

    for box in bp_teaser["boxes"]:
        box.set(facecolor="lightblue", alpha=0.7)
    for box in bp_goicp["boxes"]:
        box.set(facecolor="lightgreen", alpha=0.7)
    for box in bp_icp["boxes"]:
        box.set(facecolor="salmon", alpha=0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(overlaps)
    ax.set_xlabel("Overlap (%)")
    ax.set_ylabel("Translation Error (meters)")
    ax.set_yscale("log")
    ax.set_title("Translation Error vs. Overlap")
    ax.grid(True, which="both", ls="--")

    # Create a manual legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="lightblue", marker="s", linestyle="", label="TEASER++"),
        Line2D([0], [0], color="lightgreen", marker="s", linestyle="", label="Go-ICP"),
        Line2D([0], [0], color="salmon", marker="s", linestyle="", label="ICP"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.show()


def teaser_experiment(n_points, overlaps, n_runs=10):
    errors_teaser_rot = {ov: [] for ov in overlaps}
    errors_teaser_trans = {ov: [] for ov in overlaps}

    for ov in overlaps:
        for _ in range(n_runs):
            base_cloud = load_bunny(N=n_points)
            A, B, R_gt, t_gt = generate_partial_overlap_clouds(base_cloud, ov)
            A_cor, B_cor = generate_correspondences(A, B)

            # TEASER++
            R_est_t, t_est_t = teaser_registration(A_cor, B_cor)
            err_rot_t = rotation_error_deg(R_est_t, R_gt)
            err_trans_t = translation_error(t_est_t, t_gt)
            errors_teaser_rot[ov].append(err_rot_t)
            errors_teaser_trans[ov].append(err_trans_t)

    return errors_teaser_rot, errors_teaser_trans


def plot_teaser_errors(errors_teaser_rot, errors_teaser_trans):
    overlaps = sorted(errors_teaser_rot.keys())
    positions = np.arange(len(overlaps))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Rotation Error Plot
    axes[0].boxplot(
        [errors_teaser_rot[ov] for ov in overlaps],
        positions=positions,
        widths=0.3,
        patch_artist=True,
        showfliers=False,
    )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(overlaps)
    axes[0].set_xlabel("Overlap (%)")
    axes[0].set_ylabel("Rotation Error (deg)")
    axes[0].set_yscale("log")
    axes[0].set_title("TEASER++ Rotation Error vs. Overlap")
    axes[0].grid(True, which="both", ls="--")
    axes[0].invert_xaxis()

    # Translation Error Plot
    axes[1].boxplot(
        [errors_teaser_trans[ov] for ov in overlaps],
        positions=positions,
        widths=0.3,
        patch_artist=True,
        showfliers=False,
    )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(overlaps)
    axes[1].set_xlabel("Overlap (%)")
    axes[1].set_ylabel("Translation Error (meters)")
    axes[1].set_yscale("log")
    axes[1].set_title("TEASER++ Translation Error vs. Overlap")
    axes[1].grid(True, which="both", ls="--")
    axes[1].invert_xaxis()

    plt.tight_layout()
    plt.show()


def main():
    # base_cloud = load_bunny(N=100)
    overlaps = [90, 70, 50, 30, 10]  # Experiment with different overlaps
    n_runs = 10
    n_points = 120

    if True:
        errors_teaser_rot, errors_teaser_trans = teaser_experiment(
            n_points, overlaps, n_runs=n_runs
        )

        plot_teaser_errors(errors_teaser_rot, errors_teaser_trans)

    if False:
        (
            errors_teaser_rot,
            errors_goicp_rot,
            errors_icp_rot,
            errors_teaser_trans,
            errors_goicp_trans,
            errors_icp_trans,
        ) = rotation_experiment(base_cloud, overlaps, n_runs=n_runs)

        plot_errors(
            errors_teaser_rot,
            errors_goicp_rot,
            errors_icp_rot,
            errors_teaser_trans,
            errors_goicp_trans,
            errors_icp_trans,
        )


if __name__ == "__main__":
    main()
