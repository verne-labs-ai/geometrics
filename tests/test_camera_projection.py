import unittest
import numpy as np

from geometrics.camera.projection import (
    create_point_cloud_from_depth,
    create_point_cloud_from_depth_and_color,
)


class TestCameraProjection(unittest.TestCase):
    def setUp(self):
        # Simple 2×2 depth map and RGB image
        self.depth = np.array([[1.0, 2.0],
                               [0.0, 3.0]])
        self.rgb = np.array([
            [[255,   0,   0], [  0, 255,   0]],
            [[  0,   0, 255], [255, 255, 255]],
        ], dtype=np.uint8)
        # Identity intrinsics
        self.intrinsics = np.eye(3)

    def test_create_point_cloud_from_depth_mask_zeros_true(self):
        # mask_zeros=True → drop zero-depth pixel
        pc = create_point_cloud_from_depth(self.depth, self.intrinsics, mask_zeros=True)
        # Should yield 3 points
        self.assertEqual(pc.shape, (3, 3))
        # First point: pixel (0,0) depth=1 → [0,0,1]
        np.testing.assert_allclose(pc[0], [0.0, 0.0, 1.0])

    def test_create_point_cloud_from_depth_mask_zeros_false(self):
        # mask_zeros=False → include zero-depth
        pc = create_point_cloud_from_depth(self.depth, self.intrinsics, mask_zeros=False)
        # Should yield 4 points
        self.assertEqual(pc.shape, (4, 3))
        # One of them must be [0,0,0] (for the zero-depth pixel)
        self.assertTrue(any(np.allclose(pt, [0.0, 0.0, 0.0]) for pt in pc))

    def test_create_point_cloud_from_depth_and_color(self):
        # mask_zeros=True again → 3 points, each with [x,y,z,r,g,b]
        pc_color = create_point_cloud_from_depth_and_color(
            self.depth, self.rgb, self.intrinsics, mask_zeros=True
        )
        self.assertEqual(pc_color.shape, (3, 6))
        # First row: xyz [0,0,1], rgb normalized [1,0,0]
        first = pc_color[0]
        np.testing.assert_allclose(first[:3], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(first[3:], [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
