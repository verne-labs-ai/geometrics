import torch
from time import time
import numpy as np
import open3d as o3d
import fpsample
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


def filter_points_in_sphere(points, center, radius):
    """
    Filters out points that lie outside a sphere of given radius centered at `center`.

    Parameters
    ----------
    points : array-like, shape (N, 3)
        A list or array of 3D points.
    center : array-like, shape (3,)
        The center of the sphere.
    radius : float
        The radius of the sphere.

    Returns
    -------
    filtered_points : numpy.ndarray, shape (M, 3)
        An array of the points that lie inside or on the sphere.
    """

    # Convert inputs to numpy arrays for easy vectorized computation
    points = np.array(points, dtype=float)
    center = np.array(center, dtype=float)

    # Compute Euclidean distance of each point from the center
    distances = np.linalg.norm(points - center, axis=1)

    # Create a boolean mask for points within the sphere
    mask = distances <= radius

    # Filter the points
    filtered_points = points[mask]

    # Return points as a numpy array.
    # If you need a Python list, you can use `filtered_points.tolist()`
    return filtered_points


def remove_outliers(xyzrgb, nb_neighbors=15, std_ratio=0.5, print_progress=False):
    # remove outliers
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzrgb[:, :3])
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio, print_progress)
    xyzrgb = xyzrgb[ind]
    return xyzrgb


def filter_points_by_bounds(points, bounds_min, bounds_max, strict=True):
    """
    Filter points by taking only points within workspace bounds.
    """
    assert points.shape[1] == 3, "points must be (N, 3)"
    bounds_min = bounds_min.copy()
    bounds_max = bounds_max.copy()
    if not strict:
        bounds_min[:2] = bounds_min[:2] - 0.1 * (bounds_max[:2] - bounds_min[:2])
        bounds_max[:2] = bounds_max[:2] + 0.1 * (bounds_max[:2] - bounds_min[:2])
        bounds_min[2] = bounds_min[2] - 0.1 * (bounds_max[2] - bounds_min[2])
    within_bounds_mask = (
        (points[:, 0] >= bounds_min[0])
        & (points[:, 0] <= bounds_max[0])
        & (points[:, 1] >= bounds_min[1])
        & (points[:, 1] <= bounds_max[1])
        & (points[:, 2] >= bounds_min[2])
        & (points[:, 2] <= bounds_max[2])
    )
    return within_bounds_mask


def filter_by_distance(xyz, distance_threshold):
    return xyz[np.linalg.norm(xyz[:, :3], axis=1) < distance_threshold]


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def pc_normalize(pc):
    if type(pc).__module__ == np.__name__:
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
    else:
        centroid = torch.mean(pc, dim=0)
        pc = pc - centroid
        m = torch.max(torch.sqrt(torch.sum(pc**2, dim=1)))
        pc = pc / m
    return pc


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def farthest_point_sample_numpy(xyz, npoint):
    """
    :param xyz: point cloud data [N, 3]
    :param npoint:
    :return:
    """
    assert len(xyz) >= npoint
    indices = fpsample.bucket_fps_kdline_sampling(xyz[:, :3], npoint, h=7)
    xyz = xyz[indices]
    return xyz


def random_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        idxs: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    idxs = torch.randint(0, N, (B, npoint), dtype=torch.long).to(device)
    return idxs


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False, knn=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    if knn:
        dists = square_distance(new_xyz, xyz)  # B x npoint x N
        idx = dists.argsort()[:, :, :nsample]  # B x npoint x K
    else:
        idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def shuffle_data(data, labels):
    """Shuffle data and labels.
    Input:
      data: B,N,... numpy array
      label: B,... numpy array
    Return:
      shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Transform points using a 4x4 transformation matrix.
    Args:
        points: (N, 3) array of points
        transform: (4, 4) transformation matrix
    Returns:
        (N, 3) array of transformed points
    """
    assert points.shape[-1] == 3, f"points must be (N, 3) array, got {points.shape}"
    transformed_points = points @ transform[:3, :3].T
    transformed_points += transform[:3, 3]

    return transformed_points


def transform_points_torch(points: torch.tensor, transform: torch.tensor) -> torch.tensor:
    """
    Transform points using a 4x4 transformation matrix.
    Args:
        points: (N, 3) array of points
        transform: (4, 4) transformation matrix
    Returns:
        (N, 3) array of transformed points
    """
    output_points = points
    output_points[:, :3] = points[:, :3] @ transform[:3, :3].T
    output_points[:, :3] += transform[:3, 3]

    return output_points


def select_visible_points(
    point_cloud: np.ndarray,
    camera_pose: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
    debug: bool = False,
) -> np.ndarray:
    """
    Selects points from the original point cloud that are visible in the new camera's view and returns their 3D coordinates in world space.

    Parameters:
    original_cloud (numpy.ndarray): Nx3 array of 3D points in world coordinates.
    extrinsics (numpy.ndarray): 4x4 transformation matrix.
    intrinsics (dict): Contains 'fx', 'fy', 'cx', 'cy', 'image_width', 'image_height'.

    Returns:
    numpy.ndarray: Mx3 array of visible 3D points in world coordinates.
    """

    assert point_cloud.shape[1] == 3, "point_cloud must be (N, 3)"

    extrinsics = np.linalg.inv(camera_pose)
    camera_coords = transform_points(point_cloud, extrinsics)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    Z = camera_coords[:, 2]

    with np.errstate(divide="ignore", invalid="ignore"):
        u = (fx * camera_coords[:, 0] / Z + cx).astype(int)
        v = (fy * camera_coords[:, 1] / Z + cy).astype(int)

    # Vectorized minimum Z selection using integer indexing
    uv = np.column_stack((u, v))
    sorted_indices = np.argsort(Z)
    uv_sorted = uv[sorted_indices]

    # Find first occurrence of each (u, v) in Z-sorted order
    _, unique_indices = np.unique(uv_sorted, axis=0, return_index=True)
    selected_indices = sorted_indices[unique_indices]

    # Filter points using direct indexing
    point_cloud = point_cloud[selected_indices]
    u = u[selected_indices]
    v = v[selected_indices]
    Z = Z[selected_indices]
    camera_coords = camera_coords[selected_indices]

    # Apply visibility mask
    mask = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height) & (Z > 0)
    visible_camera = camera_coords[mask]
    visible_u = u[mask]
    visible_v = v[mask]
    visible_Z = visible_camera[:, 2]

    if debug:
        image = np.zeros((image_height, image_width))
        image[visible_v, visible_u] = 1
        plt.imshow(image)
        plt.savefig(f"{np.random.randint(0, 1000000)}_visible_points.png")
        plt.close()

    inv_fx, inv_fy = 1.0 / fx, 1.0 / fy
    X_cam = (visible_u - cx) * visible_Z * inv_fx
    Y_cam = (visible_v - cy) * visible_Z * inv_fy
    reproj_camera = np.column_stack((X_cam, Y_cam, visible_Z))

    homogeneous_reproj = np.column_stack((reproj_camera, np.ones(len(reproj_camera))))
    reproj_world = (homogeneous_reproj @ camera_pose.T)[:, :3]

    return reproj_world


def filter_point_cloud(pcd: np.ndarray, n_points: int) -> np.ndarray:
    # obj_pcd = remove_outliers(pcd)
    obj_pcd = filter_outliers_by_std_div(pcd, threshold=2)
    sampled_indices = fpsample.bucket_fps_kdline_sampling(obj_pcd[:, :3], n_points, h=7)
    return obj_pcd[sampled_indices]


def filter_outliers_by_std_div(pcd: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    mean = np.mean(pcd, axis=0)
    std_dev = np.std(pcd, axis=0)

    x_mask = np.abs(pcd[:, 0] - mean[0]) <= threshold * std_dev[0]
    y_mask = np.abs(pcd[:, 1] - mean[1]) <= threshold * std_dev[1]
    z_mask = np.abs(pcd[:, 2] - mean[2]) <= threshold * std_dev[2]

    return pcd[x_mask & y_mask & z_mask]


def count_point_cloud_overlap(
    scene_points: np.ndarray, object_points: np.ndarray, radius: float
) -> int:
    """
    Count how many points in 'object_points' lie within 'radius' of
    any point in 'scene_points' using a cKDTree for efficient lookup.

    Parameters
    ----------
    scene_points : np.ndarray
        A (num_scene_points, 3) array of 3D points representing the scene.
    object_points : np.ndarray
        A (num_object_points, 3) array of 3D points representing the object.
    radius : float
        The distance threshold for considering a point in the object
        to overlap with the scene.

    Returns
    -------
    int
        The number of points in 'object_points' that have at least one
        corresponding scene point within 'radius'.
    """
    scene_tree = cKDTree(scene_points)
    distances, _ = scene_tree.query(object_points, k=1)
    count_within_radius = np.sum(distances <= radius)

    return count_within_radius


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
    # Build a KD-tree for the object point cloud for fast nearest neighbor queries.
    tree = cKDTree(object_pc[:, :3])
    
    # Query the KD-tree for the distance from each scene point to the nearest object point.
    distances, _ = tree.query(scene_pc[:, :3], k=1)
    
    # Create a boolean mask where True indicates scene points that are further away than the threshold.
    mask = distances > threshold
    
    # Filter out the scene points that are within the threshold distance.
    filtered_scene = scene_pc[mask]
    
    return filtered_scene
