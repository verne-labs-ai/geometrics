import mink
import numpy as np


def interpolate_se3(T1: np.ndarray, T2: np.ndarray, t: float) -> np.ndarray:
    """SE(3) interpolation using Lie algebra.

    Produces screw motions (simultaneous rotation and translation along a
    common axis) rather than separate SLERP + linear interpolation.

    Formula: T(t) = T1 @ exp(t * log(T1^{-1} @ T2))

    Args:
        T1: Starting 4x4 transformation matrix.
        T2: Ending 4x4 transformation matrix.
        t: Interpolation parameter in [0, 1].

    Returns:
        Interpolated 4x4 transformation matrix.
    """
    SE3_1 = mink.SE3.from_matrix(T1)
    SE3_2 = mink.SE3.from_matrix(T2)

    T_rel = SE3_1.inverse() @ SE3_2
    twist = T_rel.log()

    T_interp = SE3_1 @ mink.SE3.exp(t * twist)

    return T_interp.as_matrix()


def interpolate_se3_trajectory(
    poses: np.ndarray,
    num_points: int,
) -> np.ndarray:
    """Interpolate SE(3) trajectory using Lie algebra interpolation.

    Uses arc-length parameterization for even spacing, with proper
    screw motion interpolation between keyframes.

    Args:
        poses: Array of shape (N, 4, 4) with SE(3) transformations.
        num_points: Desired number of output poses.

    Returns:
        Interpolated poses of shape (num_points, 4, 4).
    """
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("poses must have shape (N, 4, 4)")

    n_input = len(poses)
    if n_input == 0 or num_points <= 0:
        return poses[:0]
    if num_points == 1 or n_input == 1:
        return poses[:1]

    positions = poses[:, :3, 3]
    diffs = np.diff(positions, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_lens = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_lens[-1]

    if total_len == 0.0:
        return np.repeat(poses[:1], num_points, axis=0)

    distances = np.linspace(0.0, total_len, num_points)

    result = []
    seg_idx = 0

    for d in distances:
        while seg_idx < len(seg_lens) - 1 and d > cum_lens[seg_idx + 1]:
            seg_idx += 1

        seg_len = seg_lens[seg_idx]
        t = 0.0 if seg_len == 0 else (d - cum_lens[seg_idx]) / seg_len

        T_interp = interpolate_se3(poses[seg_idx], poses[seg_idx + 1], t)
        result.append(T_interp)

    return np.stack(result, axis=0)
