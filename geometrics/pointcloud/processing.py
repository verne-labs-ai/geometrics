from time import time
import numpy as np
import open3d as o3d
import fpsample
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


def filter_points_in_sphere(points, center, radius):
    """
    Filters out points that lie outside a sphere of given radius centered at `center`.

    Parameters:
        points (np.ndarray): Array of shape (N, 3) representing 3D points.
        center (np.ndarray): Array of shape (3,) for the sphere center.
        radius (float): Radius of the sphere.

    Returns:
        np.ndarray: Array of shape (M, 3) containing points within or on the sphere.
    """
    points = np.array(points, dtype=float)
    center = np.array(center, dtype=float)
    distances = np.linalg.norm(points - center, axis=1)
    mask = distances <= radius
    return points[mask]


def remove_statistical_outlier(xyzrgb, nb_neighbors=15, std_ratio=0.5, print_progress=False):
    """
    Remove statistical outliers from a colored point cloud.

    Parameters:
        xyzrgb (np.ndarray): Array of shape (N, >=3), where columns 0–2 are XYZ.
        nb_neighbors (int): Number of neighbors to analyze for each point.
        std_ratio (float): Standard deviation ratio threshold.
        print_progress (bool): Whether to print progress messages.

    Returns:
        np.ndarray: Subset of `xyzrgb` after removing statistical outliers.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzrgb[:, :3])
    _, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio, print_progress)
    return xyzrgb[ind]


def filter_points_by_bounds(points, bounds_min, bounds_max, strict=True):
    """
    Generate a mask for points within workspace bounds.

    Parameters:
        points (np.ndarray): Array of shape (N, 3).
        bounds_min (array-like): Minimum [x, y, z] coordinates.
        bounds_max (array-like): Maximum [x, y, z] coordinates.
        strict (bool): If False, expands XY bounds by 10% (and Z by 10% downward).

    Returns:
        np.ndarray: Boolean mask of shape (N,) indicating points within bounds.
    """
    assert points.shape[1] == 3, f"points must be (N, 3), got {points.shape}"
    bmin = np.array(bounds_min, dtype=float).copy()
    bmax = np.array(bounds_max, dtype=float).copy()
    if not strict:
        bmin[:2] -= 0.1 * (bmax[:2] - bmin[:2])
        bmax[:2] += 0.1 * (bmax[:2] - bmin[:2])
        bmin[2] -= 0.1 * (bmax[2] - bmin[2])
    mask = (
        (points[:, 0] >= bmin[0]) & (points[:, 0] <= bmax[0]) &
        (points[:, 1] >= bmin[1]) & (points[:, 1] <= bmax[1]) &
        (points[:, 2] >= bmin[2]) & (points[:, 2] <= bmax[2])
    )
    return mask


def filter_by_distance(xyz, distance_threshold):
    """
    Filter points closer than a given threshold to the origin.

    Parameters:
        xyz (np.ndarray): Array of shape (N, >=3), using the first three columns as XYZ.
        distance_threshold (float): Distance cutoff.

    Returns:
        np.ndarray: Subset of `xyz` where distance < `distance_threshold`.
    """
    dists = np.linalg.norm(xyz[:, :3], axis=1)
    return xyz[dists < distance_threshold]


def timeit(tag, t):
    """
    Print and return elapsed time since a start timestamp.

    Parameters:
        tag (str): Label for the timer.
        t (float): Start time (as returned by time.time()).

    Returns:
        float: New timestamp (time at function exit).
    """
    elapsed = time() - t
    print(f"{tag}: {elapsed}s")
    return time()


def farthest_point_sample_numpy(xyz, npoint):
    """
    Sample points via farthest-point sampling (FPS) using a KD-tree method.

    Parameters:
        xyz (np.ndarray): Array of shape (N, 3).
        npoint (int): Number of points to sample.

    Returns:
        np.ndarray: Array of shape (npoint, 3) of sampled points.
    """
    assert len(xyz) >= npoint, "Not enough points to sample."
    indices = fpsample.bucket_fps_kdline_sampling(xyz[:, :3], npoint, h=7)
    return xyz[indices]


def shuffle_data(data, labels):
    """
    Shuffle data and labels synchronously.

    Parameters:
        data (np.ndarray): Array whose first dimension matches `labels`.
        labels (np.ndarray): 1D array of labels.

    Returns:
        tuple:
            np.ndarray: Shuffled data.
            np.ndarray: Shuffled labels.
            np.ndarray: Permutation indices.
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Apply a 4×4 homogeneous transformation to 3D points.

    Parameters:
        points (np.ndarray): Array of shape (N, 3).
        transform (np.ndarray): Array of shape (4, 4).

    Returns:
        np.ndarray: Transformed points of shape (N, 3).
    """
    assert points.shape[-1] == 3, f"points must be (N, 3), got {points.shape}"
    pts = points @ transform[:3, :3].T
    pts += transform[:3, 3]
    return pts


def select_visible_points(
    point_cloud: np.ndarray,
    camera_pose: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
    debug: bool = False,
) -> np.ndarray:
    """
    Select points visible in a camera view and return their 3D world coordinates.

    Parameters:
        point_cloud (np.ndarray): Array of shape (N, 3) in world coordinates.
        camera_pose (np.ndarray): 4×4 camera-to-world transformation.
        intrinsics (np.ndarray): 3×3 intrinsic matrix.
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.
        debug (bool): If True, save a debug visibility mask image.

    Returns:
        np.ndarray: Array of visible world points of shape (M, 3).
    """
    assert point_cloud.shape[1] == 3, f"point_cloud must be (N, 3), got {point_cloud.shape}"

    # Transform to camera frame
    extrinsics = np.linalg.inv(camera_pose)
    cam_pts = transform_points(point_cloud, extrinsics)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    Z = cam_pts[:, 2]

    with np.errstate(divide="ignore", invalid="ignore"):
        u = (fx * cam_pts[:, 0] / Z + cx).astype(int)
        v = (fy * cam_pts[:, 1] / Z + cy).astype(int)

    # Depth ordering and uniqueness per pixel
    uv = np.column_stack((u, v))
    order = np.argsort(Z)
    uv_sorted = uv[order]
    _, unique_idx = np.unique(uv_sorted, axis=0, return_index=True)
    sel = order[unique_idx]

    # Apply image bounds and positive depth
    u_sel, v_sel, Z_sel = u[sel], v[sel], Z[sel]
    mask = (u_sel >= 0) & (u_sel < image_width) & (v_sel >= 0) & (v_sel < image_height) & (Z_sel > 0)
    vis_cam = cam_pts[sel][mask]

    if debug:
        img = np.zeros((image_height, image_width))
        coords = np.column_stack((v_sel[mask], u_sel[mask]))
        img[coords[:, 0], coords[:, 1]] = 1
        plt.imshow(img)
        plt.savefig(f"{np.random.randint(1e6)}_visible_points.png")
        plt.close()

    # Reproject back to world
    inv_fx, inv_fy = 1.0 / fx, 1.0 / fy
    u_vis = u_sel[mask]
    v_vis = v_sel[mask]
    z_vis = vis_cam[:, 2]
    x_cam = (u_vis - cx) * z_vis * inv_fx
    y_cam = (v_vis - cy) * z_vis * inv_fy
    reproj_cam = np.column_stack((x_cam, y_cam, z_vis))
    homo = np.column_stack((reproj_cam, np.ones(len(reproj_cam))))
    return (homo @ camera_pose.T)[:, :3]


def filter_point_cloud(pcd: np.ndarray, n_points: int) -> np.ndarray:
    """
    Remove outliers by standard-deviation threshold and sample points.

    Parameters:
        pcd (np.ndarray): Array of shape (N, >=3).
        n_points (int): Number of points to sample.

    Returns:
        np.ndarray: Filtered and sampled point cloud of shape (n_points, >=3).
    """
    obj = filter_outliers_by_std_div(pcd, threshold=2)
    idx = fpsample.bucket_fps_kdline_sampling(obj[:, :3], n_points, h=7)
    return obj[idx]


def filter_outliers_by_std_div(pcd: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Remove points lying beyond a multiple of the standard deviation.

    Parameters:
        pcd (np.ndarray): Array of shape (N, 3).
        threshold (float): Number of standard deviations for cutoff.

    Returns:
        np.ndarray: Subset of `pcd` within `threshold` standard deviations.
    """
    mean = np.mean(pcd, axis=0)
    std = np.std(pcd, axis=0)
    mask = (
        (np.abs(pcd[:, 0] - mean[0]) <= threshold * std[0]) &
        (np.abs(pcd[:, 1] - mean[1]) <= threshold * std[1]) &
        (np.abs(pcd[:, 2] - mean[2]) <= threshold * std[2])
    )
    return pcd[mask]


def count_point_cloud_overlap(scene_points: np.ndarray, object_points: np.ndarray, radius: float) -> int:
    """
    Count object points overlapping scene points within a given radius.

    Parameters:
        scene_points (np.ndarray): Array of shape (Ns, 3).
        object_points (np.ndarray): Array of shape (No, 3).
        radius (float): Distance threshold for overlap.

    Returns:
        int: Number of object points whose nearest scene point is within `radius`.
    """
    tree = cKDTree(scene_points)
    dists, _ = tree.query(object_points, k=1)
    return int(np.sum(dists <= radius))


def remove_object_from_scene(scene_pc: np.ndarray, object_pc: np.ndarray, threshold: float) -> np.ndarray:
    """
    Remove object points from a scene point cloud using distance matching.

    Parameters:
        scene_pc (np.ndarray): The scene point cloud as an array of shape (N, 3).
        object_pc (np.ndarray): The object point cloud as an array of shape (M, 3).
        threshold (float): Distance threshold for matching; scene points closer than this 
                           to any object point will be removed.

    Returns:
        np.ndarray: The filtered scene point cloud with object points removed.
    """
    tree = cKDTree(object_pc[:, :3])
    dists, _ = tree.query(scene_pc[:, :3], k=1)
    mask = dists > threshold
    return scene_pc[mask]
