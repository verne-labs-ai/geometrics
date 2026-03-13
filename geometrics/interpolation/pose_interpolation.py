import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from geometrics.transforms.rigid import quaternion_from_matrix, quat_slerp


def interpolate_line(y_values, kind="linear"):
    # x-values are the integer indices of the y-values
    x_values = np.arange(len(y_values))
    # Create an interpolation function
    interp_function = interp1d(x_values, y_values, kind=kind, fill_value="extrapolate")
    return interp_function


def interpolate_poses(pos1, rot1, pos2, rot2, num_steps):
    """
    Linear interpolation between two poses.
    Args:
        pos1 (np.array): np array of shape (3,) for first position
        rot1 (np.array): np array of shape (3, 3) for first rotation
        pos2 (np.array): np array of shape (3,) for second position
        rot2 (np.array): np array of shape (3, 3) for second rotation
        num_steps (int): specifies the number of desired interpolated points (not including
            the start and end points). Passing 0 corresponds to no interpolation.
    Returns:
        pos_steps (np.array): array of shape (N + 2, 3) corresponding to the interpolated position path, where N is @num_steps
        rot_steps (np.array): array of shape (N + 2, 4) corresponding to the interpolated rotation path, where N is @num_steps (xyzw convention)
        num_steps (int): the number of interpolated points (N) in the path
    """
    if num_steps == 0:
        # skip interpolation
        return (
            np.concatenate([pos1[None], pos2[None]], axis=0),
            np.concatenate([rot1[None], rot2[None]], axis=0),
            num_steps,
        )
    delta_pos = pos2 - pos1
    num_steps += 1  # include starting pose
    assert num_steps >= 2
    # linear interpolation of positions
    pos_step_size = delta_pos / num_steps
    grid = np.arange(num_steps).astype(np.float64)
    pos_steps = np.array([pos1 + grid[i] * pos_step_size for i in range(num_steps)])
    # add in endpoint
    pos_steps = np.concatenate([pos_steps, pos2[None]], axis=0)
    # interpolate the rotations too
    rot_steps = interpolate_rotations(R1=rot1, R2=rot2, num_steps=num_steps)
    return pos_steps, rot_steps, num_steps - 1


def interpolate_rotations(R1: np.ndarray, R2: np.ndarray, num_steps: int):
    """
    Interpolate between 2 rotation matrices.
    Return a list of quaternions (xyzw) representing the interpolated rotations.
    """
    
    tmp = np.eye(4)
    tmp[:3, :3] = R1
    R1 = tmp
    tmp = np.eye(4)
    tmp[:3, :3] = R2
    R2 = tmp

    q1 = quaternion_from_matrix(R1)
    q2 = quaternion_from_matrix(R2)
    rot_steps = np.array(
        [quat_slerp(q1, q2, fraction=(float(i) / num_steps)) for i in range(num_steps)]
    )

    # add in endpoint
    rot_steps = np.concatenate([rot_steps, q2[None]], axis=0)

    return rot_steps


def se3_to_rpy(pose_4x4: np.ndarray):
    """
    Convert a 4x4 homogeneous transform to (x, y, z, roll, pitch, yaw).
    pose_4x4 is assumed to be of the form:
       [ R  t ]
       [ 0  1 ]
    where R is a 3x3 rotation and t is a 3x1 translation.
    """
    # Extract translation
    x, y, z = pose_4x4[:3, 3]

    # Extract rotation and convert to roll, pitch, yaw (XYZ convention or similar)
    R_mat = pose_4x4[:3, :3]  # 3x3 rotation
    rpy = R.from_matrix(R_mat).as_euler('xyz', degrees=False)  # roll, pitch, yaw
    roll, pitch, yaw = rpy

    return x, y, z, roll, pitch, yaw

def rpy_to_se3(x: float, y: float, z: float,
               roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert (x, y, z, roll, pitch, yaw) to a 4x4 homogeneous transform.
    """
    # Create rotation from Euler
    R_mat = R.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()

    # Build 4x4
    pose_4x4 = np.eye(4)
    pose_4x4[:3, :3] = R_mat
    pose_4x4[:3, 3]  = [x, y, z]

    return pose_4x4


def interpolate_poses_euler(poses: np.ndarray, N: int = 20) -> np.ndarray:
    """
    Given a list of 4x4 SE(3) poses, interpolate them into N new poses
    (total of N samples across the entire sequence).

    Args:
        poses: list of 4x4 numpy arrays representing SE(3) transforms.
        N: number of interpolated points across the entire sequence.

    Returns:
        new_poses: list of 4x4 numpy arrays (length = N).
    """
    n = len(poses)
    if n < 2:
        raise ValueError("Need at least two poses to interpolate.")

    # 1) Extract x, y, z, roll, pitch, yaw for each pose
    xyz_rpy = np.array([se3_to_rpy(p) for p in poses])  # shape (n, 6)
    x_vals   = xyz_rpy[:, 0]
    y_vals   = xyz_rpy[:, 1]
    z_vals   = xyz_rpy[:, 2]
    roll_vals  = xyz_rpy[:, 3]
    pitch_vals = xyz_rpy[:, 4]
    yaw_vals   = xyz_rpy[:, 5]

    # (Optional) "unwrap" angles to reduce discontinuities. 
    # If rotations do not exceed 2π jumps, this can help keep them continuous:
    roll_vals  = np.unwrap(roll_vals)
    pitch_vals = np.unwrap(pitch_vals)
    yaw_vals   = np.unwrap(yaw_vals)

    # 2) Parameter t for the given poses: 0, 1, 2, ..., n-1
    t_original = np.arange(n)

    # 3) Create a new time vector for interpolation
    t_new = np.linspace(0, n-1, N)

    # 4) Use cubic splines for each dimension
    x_spline    = CubicSpline(t_original, x_vals)
    y_spline    = CubicSpline(t_original, y_vals)
    z_spline    = CubicSpline(t_original, z_vals)
    roll_spline = CubicSpline(t_original, roll_vals)
    pitch_spline= CubicSpline(t_original, pitch_vals)
    yaw_spline  = CubicSpline(t_original, yaw_vals)

    # 5) Sample the splines
    x_new    = x_spline(t_new)
    y_new    = y_spline(t_new)
    z_new    = z_spline(t_new)
    roll_new = roll_spline(t_new)
    pitch_new= pitch_spline(t_new)
    yaw_new  = yaw_spline(t_new)

    # 6) Rebuild the 4x4 poses
    new_poses = []
    for i in range(N):
        pose_4x4 = rpy_to_se3(x_new[i],
                              y_new[i],
                              z_new[i],
                              roll_new[i],
                              pitch_new[i],
                              yaw_new[i])
        new_poses.append(pose_4x4)

    return np.array(new_poses)


def interpolate_joint_angles(joint_angle_list, threshold):
    """
    Interpolate between a list of 7-DOF joint angles such that:
      1. The distance between consecutive angles does not exceed 'threshold'.
      2. The distance between consecutive angles is kept roughly constant for uniform speed.
      3. The resulting trajectory is piecewise linearly interpolated (smooth in that sense).

    Parameters
    ----------
    joint_angle_list : np.ndarray
        Array of shape [n, 7], where each row is a 7-DOF joint configuration.
    threshold : float
        Maximum allowed distance (in Euclidean sense) between consecutive interpolated angles.

    Returns
    -------
    interpolated_path : np.ndarray
        Array of shape [m, 7], where m >= n (due to added interpolation points).
        This path satisfies the distance constraint between consecutive angles.
    """

    # Make sure input is a numpy array
    angles = np.asarray(joint_angle_list)

    # A list to collect the interpolated path
    interpolated_path = [angles[0].copy()]

    # Iterate over each consecutive pair of angles
    for i in range(len(angles) - 1):
        start = angles[i]
        end = angles[i + 1]

        # Vector difference
        diff = end - start

        # Euclidean distance between the two angles in 7D space
        dist = np.linalg.norm(diff)

        # Number of steps needed so that each step <= threshold
        num_steps = int(np.ceil(dist / threshold)) if dist > 0 else 1

        # Interpolate between start and end
        for step in range(1, num_steps + 1):
            alpha = step / num_steps
            interpolated_point = start + alpha * diff
            interpolated_path.append(interpolated_point)

    return np.array(interpolated_path)
