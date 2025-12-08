# geometrics

Modular 3D geometry utilities for robotics and computer vision.

## Installation

```bash
pip install -e .
```

Requires Python >= 3.11.

## Modules

### Types (`geometrics.types`)
Strongly-typed numpy array wrappers with shape validation:
- `Matrix3x3`, `RotationMatrix`, `Rotation6D`, `Quaternion`, `Pose4x4`

### Transforms (`geometrics.transforms`)
Rigid body transformations:
- Rotation matrices, quaternions, Euler angles
- SE(3) homogeneous transforms
- Pose conversions (mat, quat, euler, 6D, 10D)

### Camera (`geometrics.camera`)
- Point cloud generation from depth maps

### Point Cloud (`geometrics.pointcloud`)
- Filtering (bounds, sphere, distance)
- Outlier removal (statistical, radius-based)
- Farthest point sampling
- Visibility selection

### Interpolation (`geometrics.interpolation`)
- Pose interpolation (linear, SLERP, cubic spline)
- Joint angle interpolation

### Pose (`geometrics.pose`)
- Pose utilities and delta application