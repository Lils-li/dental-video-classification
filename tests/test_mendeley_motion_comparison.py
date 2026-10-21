"""Synthetic geometry/source-provenance tests; no models or dental data."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location("motion_comparison", Path(__file__).resolve().parents[1]/"src/synthetic/mendeley_motion_comparison.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class ComparisonTests(unittest.TestCase):
    def test_translation_is_depth_dependent(self):
        z=np.ones((64,128),np.float32);z[:,64:]=1.25
        u,v=np.indices(z.shape,dtype=np.float32)[1],np.indices(z.shape,dtype=np.float32)[0]
        pu,pv,zz=pilot.parallax_project(z,.004)
        self.assertNotAlmostEqual(float((pu-u)[:,:64].mean()),float((pu-u)[:,64:].mean()))
        np.testing.assert_array_equal(pv,v)
        self.assertGreater(pilot.parallax_metrics(z,.004)["near_far_differential_px"],0)
        self.assertEqual(pilot.parallax_metrics(np.ones_like(z),.004)["depth_vs_flat_p95_px"],0)

    def test_trajectory_monotonic_without_oscillation(self):
        samples=np.array([pilot.smooth(i) for i in range(48)])
        self.assertEqual(samples[0],0);self.assertEqual(samples[-1],1)
        self.assertTrue((np.diff(samples)>0).all())
        self.assertLess(np.diff(samples).max(),.033)

    def test_handheld_identity_and_bounded_motion(self):
        matrix,pose=pilot.handheld_homography((64,128),0)
        np.testing.assert_allclose(matrix,np.eye(3),atol=1e-10)
        final,pose=pilot.handheld_homography((64,128),47)
        self.assertAlmostEqual(pose["zoom"],1.012)
        self.assertAlmostEqual(pose["roll_degrees"],-.22)
        rgb=np.full((64,128,3),200,np.uint8)
        frame,metrics=pilot.render(rgb,None,"handheld",0)
        np.testing.assert_array_equal(frame,rgb)
        self.assertEqual(metrics["hole_fraction"],0)

    def test_largest_passing_candidate_and_failed_fallback(self):
        def fake_render(rgb,z,mode,index,amp):
            fail=amp>.001
            return None,dict(hole_fraction=.04 if fail else 0,interior_hole_fraction=0,
                            foldover_fraction=0,extreme_stretch_fraction=0,
                            failed_checks=["hole_fraction"] if fail else [],
                            pose=dict(depth_vs_flat_p95_px=amp*1000,near_far_differential_px=amp*2000))
        with patch.object(pilot,"render",side_effect=fake_render):
            chosen,trials,passed=pilot.evaluate_amplitudes(None,None)
            self.assertEqual(chosen,.001);self.assertTrue(passed);self.assertEqual(len(trials),4)
            self.assertTrue(all(len(t["per_frame"])==48 for t in trials))
            chosen,_,passed=pilot.evaluate_amplitudes(None,None,[.002,.004])
            self.assertEqual(chosen,.002);self.assertFalse(passed)

    def test_sources_match_saved_metadata_and_reject_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/"data";yaw=root/"yaw";data.mkdir();yaw.mkdir()
            for i in range(1,6):
                source=data/f"image{i}.jpg";source.write_bytes(f"synthetic{i}".encode())
                folder=yaw/f"photo_{i:02d}";folder.mkdir()
                (folder/"metadata.json").write_text(json.dumps(dict(source_relative_path=source.name,source_sha256=pilot.base.sha(source),model_revision="a"*40)))
            sources,revision=pilot.load_sources(yaw,data)
            self.assertEqual(len(sources),5);self.assertEqual(revision,"a"*40)
            (data/"image1.jpg").write_bytes(b"changed")
            with self.assertRaises(ValueError):pilot.load_sources(yaw,data)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/"data";data.mkdir();folder=root/"yaw/photo_01";folder.mkdir(parents=True)
            (root/"outside.jpg").write_bytes(b"test")
            (folder/"metadata.json").write_text(json.dumps(dict(source_relative_path="../outside.jpg",source_sha256=pilot.base.sha(root/"outside.jpg"),model_revision="a"*40)))
            with self.assertRaises(ValueError):pilot.load_sources(root/"yaw",data)


if __name__=="__main__":unittest.main()
