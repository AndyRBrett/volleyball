import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pipeline  # noqa: E402


class TestPipelineSelfTest(unittest.TestCase):
    """The end-to-end guard: run the full CV pipeline on the bundled reference
    clip. If detection, segmentation, tagging, or coaching breaks, this fails
    the build instead of letting the pipeline sit silently at zero frames."""

    def test_self_test_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = pipeline.self_test(results_dir=tmp, verbose=False)

            tracking = result["tracking"]
            report = result["report"]
            manifest = result["manifest"]

            # Detection actually processed frames (the whole point).
            self.assertGreater(tracking["frame_count"], 0)
            self.assertGreater(tracking["detected_frames"], 0)

            # Highlights + coaching produced real output.
            self.assertGreaterEqual(manifest["rally_count"], 1)
            self.assertEqual(report["rally_count"], manifest["rally_count"])
            self.assertTrue(any(r["tags"] for r in report["rallies"]))
            self.assertGreater(report["contact_heatmap"]["contacts_binned"], 0)
            self.assertTrue(any(r["ball_speed"] for r in report["rallies"]))

            # Metrics roll-up is consistent (this is what write_status reads).
            metrics = result["metrics"]
            self.assertEqual(metrics["frames_processed"], tracking["frame_count"])
            self.assertEqual(metrics["footage_processed"], 1)

            # selftest.json was written for the overseer.
            self.assertTrue(os.path.exists(os.path.join(tmp, "selftest.json")))

    def test_reference_clip_has_three_rallies(self):
        # Locks in the structure of the bundled fixture so a fixture or
        # segmentation regression is caught explicitly, not just "rally_count>=1".
        with tempfile.TemporaryDirectory() as tmp:
            result = pipeline.self_test(results_dir=tmp, verbose=False)
        self.assertEqual(result["report"]["rally_count"], 3)


if __name__ == "__main__":
    unittest.main()
