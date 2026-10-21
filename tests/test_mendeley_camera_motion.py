"""Synthetic-array checks only: no photos, models, downloads or inference."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

spec = importlib.util.spec_from_file_location("camera_pilot", Path(__file__).resolve().parents[1]/"src/synthetic/mendeley_camera_motion.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class CameraTests(unittest.TestCase):
    def test_identity_and_non_square_dimensions(self):
        rgb = np.random.default_rng(42).integers(0, 256, (64, 128, 3), dtype=np.uint8)
        z = np.ones((64, 128), np.float32)
        pu, pv, zz = pilot.project(z, 0)
        frame, holes = pilot.splat(rgb, pu, pv, zz)
        np.testing.assert_array_equal(frame, rgb)
        self.assertFalse(holes.any())
        self.assertEqual(pilot.failures(pilot.artifact_metrics(pu, pv, holes)), [])

    def test_yaw_is_smooth_bounded_and_returns_to_zero(self):
        angles = np.array([pilot.yaw_degrees(i) for i in range(48)])
        self.assertLessEqual(abs(angles).max(), .5)
        self.assertAlmostEqual(angles[0], 0)
        self.assertAlmostEqual(angles[-1], 0)
        self.assertLess(np.abs(np.diff(angles)).max(), .07)
        self.assertGreater(angles.max(), .49)
        self.assertLess(angles.min(), -.49)

    def test_pure_yaw_has_no_depth_parallax(self):
        z = np.linspace(.8, 1.2, 64*128, dtype=np.float32).reshape(64, 128)
        a = pilot.project(z, .5)
        b = pilot.project(np.ones_like(z), .5)
        np.testing.assert_allclose(a[0], b[0], atol=2e-5)
        np.testing.assert_allclose(a[1], b[1], atol=2e-5)

    def test_holes_are_gray_not_inpainted(self):
        rgb = np.full((64, 128, 3), 220, np.uint8)
        pu, pv, zz = pilot.project(np.ones((64, 128), np.float32), .5)
        frame, holes = pilot.splat(rgb, pu, pv, zz)
        self.assertTrue(holes.any())
        self.assertTrue((frame[holes] == 127).all())
        self.assertLess(holes.mean(), .025)

    def test_foldovers_and_interior_holes_fail(self):
        v, u = np.indices((64, 128), dtype=np.float32)
        holes = np.zeros((64, 128), bool)
        self.assertIn("foldover_fraction", pilot.failures(pilot.artifact_metrics(-u, v, holes)))
        holes[20:30, 30:40] = True
        self.assertIn("interior_hole_fraction", pilot.failures(pilot.artifact_metrics(u, v, holes)))

    def test_selection_is_five_classes_with_unique_bytes(self):
        root = MagicMock()
        folders = []
        for name in ["Lower Front", "Upper Front", "Lower Left", "Upper Right", "Lower Occlusal"]:
            folder = MagicMock(); folder.name = name; folder.is_dir.return_value = True
            photo = MagicMock(); photo.suffix = ".JPG"
            photo.relative_to.return_value.as_posix.return_value = name + "/image.JPG"
            folder.rglob.return_value = [photo]
            folders.append(folder)
        root.iterdir.return_value = folders
        with patch.object(pilot, "sha", side_effect=lambda p: p.relative_to(root).as_posix()):
            selected = pilot.select_photos(root)
            self.assertEqual(len(selected), 5)
            self.assertEqual(len({digest for _, digest in selected}), 5)
        root.iterdir.return_value = folders
        with patch.object(pilot, "sha", return_value="duplicate"):
            with self.assertRaises(ValueError):
                pilot.select_photos(root)

    def test_invalid_geometry_rejected(self):
        z = np.ones((64, 128), np.float32)
        u, v, zz = pilot.project(z, 0)
        zz[0, 0] = np.nan
        with self.assertRaises(ValueError):
            pilot.splat(np.zeros((64, 128, 3), np.uint8), u, v, zz)


if __name__ == "__main__":
    unittest.main()
