from typing import Tuple
import numpy as np


def update_intrinsics_matrix(
    intrinsics_matrix: np.ndarray, original_shape: Tuple[int, int], new_shape: Tuple[int, int]
) -> np.ndarray:
    """_summary_

    Args:
        intrinsics_matrix (np.ndarray): _description_
        original_shape (Tuple[int, int]): _description_
        new_shape (Tuple[int, int]): _description_

    Returns:
        np.ndarray: _description_
    """

    if intrinsics_matrix.shape != (3, 3):
        raise ValueError("Intrinsics matrix must be a 3x3 matrix")
    if new_shape[0] <= 0 or new_shape[1] <= 0:
        raise ValueError("New shape must have positive width and height")
    if original_shape[0] <= 0 or original_shape[1] <= 0:
        raise ValueError("Original shape must have positive width and height")

    # Extract the focal lengths and principal points from the original matrix
    fx, fy = intrinsics_matrix[0, 0], intrinsics_matrix[1, 1]
    cx, cy = intrinsics_matrix[0, 2], intrinsics_matrix[1, 2]
    cx, cy = intrinsics_matrix[0, 2], intrinsics_matrix[1, 2]

    # Calculate the scale factors for width and height
    original_height, original_width = original_shape
    new_height, new_width = new_shape

    scale_x = new_width / original_width
    scale_y = new_height / original_height

    # Update the focal lengths and principal points
    fx_new = fx * scale_x
    fy_new = fy * scale_y
    cx_new = cx * scale_x
    cy_new = cy * scale_y

    # Create the new intrinsic matrix
    new_intrinsics_matrix = np.array([[fx_new, 0, cx_new], [0, fy_new, cy_new], [0, 0, 1]])

    return new_intrinsics_matrix


def create_point_cloud_from_depth_and_color(
    depth_map: np.ndarray, rgb_image: np.ndarray, intrinsics: np.ndarray, mask_zeros: bool = True
) -> np.ndarray:
    """
    Generate a 3D point cloud from a depth map and RGB image in the camera frame.

    Parameters:
    - depth_map (numpy.ndarray): 2D array representing the depth at each pixel.
    - rgb_image (numpy.ndarray): 3D array (H x W x 3) representing the RGB color image.
    - intrinsics (numpy.ndarray): 3x3 intrinsic matrix for the camera.

    Returns:
    - points (numpy.ndarray): N x 3 array of 3D points in the camera frame.
    - colors (numpy.ndarray): N x 3 array of RGB colors corresponding to each 3D point.
    """
    # Get image dimensions
    height, width = depth_map.shape

    # Create mesh grid for pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.flatten()
    v = v.flatten()

    # Flatten depth map and color image
    z = depth_map.flatten()
    colors = rgb_image.reshape(-1, 3)

    # Filter out points with zero depth
    if mask_zeros:
        valid_indices = z > 0
        z = z[valid_indices]
        u = u[valid_indices]
        v = v[valid_indices]
        colors = colors[valid_indices]

    # Apply the inverse of the intrinsic matrix to get normalized coordinates
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Compute X, Y, Z coordinates in the camera frame
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack to get an array of 3D points (N x 3)
    points = np.stack((x, y, z), axis=-1)

    return np.concatenate([points, colors / 255], axis=-1)


def create_point_cloud_from_depth(
    depth_map: np.ndarray, intrinsics: np.ndarray, mask_zeros: bool = True
) -> np.ndarray:
    """
    Generate a 3D point cloud from a depth map and RGB image in the camera frame.

    Parameters:
    - depth_map (numpy.ndarray): 2D array representing the depth at each pixel.
    - rgb_image (numpy.ndarray): 3D array (H x W x 3) representing the RGB color image.
    - intrinsics (numpy.ndarray): 3x3 intrinsic matrix for the camera.

    Returns:
    - points (numpy.ndarray): N x 3 array of 3D points in the camera frame.
    - colors (numpy.ndarray): N x 3 array of RGB colors corresponding to each 3D point.
    """
    # Get image dimensions
    height, width = depth_map.shape

    # Create mesh grid for pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.flatten()
    v = v.flatten()

    # Flatten depth map and color image
    z = depth_map.flatten()

    # Filter out points with zero depth
    if mask_zeros:
        valid_indices = z > 0
        z = z[valid_indices]
        u = u[valid_indices]
        v = v[valid_indices]

    # Apply the inverse of the intrinsic matrix to get normalized coordinates
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Compute X, Y, Z coordinates in the camera frame
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.stack((x, y, z), axis=-1)
