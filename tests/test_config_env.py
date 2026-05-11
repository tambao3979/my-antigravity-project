import os
import subprocess
import sys
import textwrap
import unittest


class ConfigEnvTests(unittest.TestCase):
    def test_invalid_numeric_environment_values_fall_back_without_import_crash(self):
        env = os.environ.copy()
        env.update(
            {
                "YOLO_IMGSZ": "not-an-int",
                "FIRE_CONFIRM_SECONDS": "0",
                "DISPLAY_TARGET_FPS": "-30",
                "BOX_SMOOTHING_ALPHA": "9.5",
                "MQTT_PORT": "bad-port",
            }
        )
        script = textwrap.dedent(
            """
            from config import Config
            assert Config.YOLO_IMGSZ == 960
            assert Config.FIRE_CONFIRM_SECONDS > 0
            assert Config.DISPLAY_TARGET_FPS >= 1
            assert 0.0 <= Config.BOX_SMOOTHING_ALPHA <= 1.0
            assert Config.MQTT_PORT == 1883
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_custom_fire_and_smoke_class_names_are_treated_as_hazards(self):
        env = os.environ.copy()
        env.update(
            {
                "FIRE_CLASS_NAME": "flame",
                "SMOKE_CLASS_NAME": "dark_smoke",
            }
        )
        script = textwrap.dedent(
            """
            from config import Config
            from detector import Detection

            assert "flame" in Config.HAZARD_CLASS_NAMES
            assert "dark_smoke" in Config.HAZARD_CLASS_NAMES
            assert Detection("flame", 0.90, 0, 0, 10, 10).is_fire
            assert Detection("dark_smoke", 0.80, 0, 0, 10, 10).is_fire
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
