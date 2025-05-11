import numpy as np
import torch
from scipy.spatial.transform import Rotation
from theseus.geometry import SO3

from src.utils.geometry.basic import normalize_vector, cross_product


def generate_random_poses(gripper_pose, num_poses=10, position_range=0.05, rotation_range=15):
    """
    Generate random pose matrices around a given gripper pose.

    Args:
    - gripper_pose (np.ndarray): 4x4 transformation matrix of the original gripper pose
    - num_poses (int): Number of random poses to generate
    - position_range (float): Maximum distance in meters to perturb the position
    - rotation_range (float): Maximum angle in degrees to perturb the rotation

    Returns:
    - List of 4x4 numpy arrays representing the perturbed poses
    """
    random_poses = []

    for _ in range(num_poses):
        # Random position perturbation
        position_perturbation = np.random.uniform(-position_range, position_range, 3)

        # Random rotation perturbation
        rotation_perturbation = Rotation.from_euler(
            "xyz", np.random.uniform(-rotation_range, rotation_range, 3), degrees=True
        )

        # Extract original position and rotation
        original_position = gripper_pose[:3, 3]
        original_rotation = Rotation.from_matrix(gripper_pose[:3, :3])

        # Apply perturbations
        new_position = original_position + position_perturbation
        new_rotation = original_rotation * rotation_perturbation

        # Construct new pose matrix
        new_pose = np.eye(4)
        new_pose[:3, :3] = new_rotation.as_matrix()
        new_pose[:3, 3] = new_position

        random_poses.append(new_pose)

    return random_poses


# poses batch*6
# poses
def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]  # batch*3
    y_raw = ortho6d[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


class SO3_R3:
    def __init__(self, R: torch.Tensor = None, t: torch.Tensor = None, dtype=None):
        if dtype is not None:
            self.R = SO3(dtype=dtype)
        else:
            self.R = SO3()
        if R is not None:
            self.R.update(R)
        self.w = self.R.log_map()
        if t is not None:
            self.t = t

    def log_map(self):
        return torch.cat((self.t, self.w), -1)

    def exp_map(self, x):
        self.t = x[..., :3]
        self.w = x[..., 3:]
        self.R = SO3().exp_map(self.w)
        return self

    def to_matrix(self):
        H = torch.eye(4).unsqueeze(0).repeat(self.t.shape[0], 1, 1).to(self.t)
        H[:, :3, :3] = self.R.to_matrix()
        H[:, :3, -1] = self.t
        return H

    # The quaternion takes the [w x y z] convention
    def to_quaternion(self):
        return self.R.to_quaternion()

    def sample(self, batch=1):
        R = SO3().rand(batch)
        t = torch.randn(batch, 3)
        H = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1).to(t)
        H[:, :3, :3] = R.to_matrix()
        H[:, :3, -1] = t
        return H
