import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pose_overlay  # noqa: E402
import process_footage  # noqa: E402  (ensures the new import wires up)


class TestPoseOverlay(unittest.TestCase):
    def test_module_shape(self):
        self.assertTrue(callable(pose_overlay.annotate))
        self.assertTrue(pose_overlay.DEFAULT_MODEL.endswith("-pose.pt"))

    def test_lazy_deps(self):
        # ultralytics/opencv are imported lazily inside annotate(); in an env
        # without them, annotate raises ImportError -- which process_footage
        # catches so the overlay stays best-effort. Skip if they happen to exist.
        try:
            import ultralytics  # noqa: F401
            import cv2  # noqa: F401
        except ImportError:
            with self.assertRaises(ImportError):
                pose_overlay.annotate("nonexistent.mp4")
        else:
            self.skipTest("ultralytics/opencv installed; lazy-import path not exercised")


if __name__ == "__main__":
    unittest.main()
