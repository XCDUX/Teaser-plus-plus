# Point Cloud Registration with TEASER++

## Overview
This git gathers our scripts when testing the solution for **point cloud registration** using the **TEASER++** framework, written in the context of the course **3D Point Cloud and Modeling**, Master IASD. Among others, our scripts are meant to generate synthetic point clouds, apply transformations, and estimates these transformations while handling noise and outliers. The implementation is structured into multiple Python modules, each testing a specific aspect of the registration pipeline.

## Repository Structure

- **`data.py`** - Handles data generation or loading, including synthetic point clouds and transformation parameters.
- **`rotation.py`** - Tests the estimation and evaluation of rotation transformations.
- **`scale.py`** - Tests the estimation and evaluation of scale transformations.
- **`MCIS.py`** - Tests the Maximum Clique-based Inlier Selection (MCIS) for robust correspondence filtering.
- **`translation.py`** - Tests the estimation and evaluation of translation transformations.
- **`loop.py`** - Runs the iterative registration process over a fixed number of samples so that they can be analysed, saving results and archiving failed cases.
- **`test.py`** - Contains a basic test case to track elimination of outliers.
- **`T_G_ICP.py`** - Implements TEASER++, ICP and an attempt of Generalized ICP (G-ICP) for robust registration, comparing performances.

## Installation
### Prerequisites
Ensure you have installed the TEASER++ project

## Usage
### Running the Main Loop
Execute the registration loop with:
```bash
python loop.py
```
This will generate clouds, estimate transformations, and save results in the **CSV** folder. Failed cases (where rotation estimation error is high) are archived in `CSV_k.zip`.

### Running Tests
To replicate the basic test with outlier monitoring, run:
```bash
python test.py
```

## Output Files
- **`CSV/source_cloud.csv`** - Original point cloud.
- **`CSV/target_cloud.csv`** - Transformed point cloud with noise and outliers.
- **`CSV/estimated_rotation.csv`** - Estimated rotation matrix.
- **`CSV/ground_truth_rotation.csv`** - Ground truth rotation matrix.
- **`CSV/estimated_translation.csv`** - Estimated translation vector.
- **`CSV/ground_truth_translation.csv`** - Ground truth translation vector.
- **`CSV/estimated_scale.csv`** - Estimated scale factor.
- **`CSV/ground_truth_scale.csv`** - Ground truth scale factor.
- **`CSV/max_clique_pairings_inliers.csv`** - Inlier correspondences selected by MCIS.
- **Archived Results** - If the estimated rotation differs significantly from the ground truth, results are stored in `CSV_k.zip`.

## Contributing
Feel free to open issues or submit pull requests to improve the project.

