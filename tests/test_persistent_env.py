import os
import tempfile
import unittest

from persistent_env import (
    get_persistent_env_path,
    load_application_env,
    save_env_value,
)


class PersistentEnvTests(unittest.TestCase):
    def test_get_persistent_env_path_uses_appdata_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = get_persistent_env_path(app_name="CameraAI", appdata_root=temp_dir)

        self.assertEqual(
            env_path,
            os.path.join(temp_dir, "CameraAI", ".env"),
        )

    def test_save_env_value_updates_existing_key_and_keeps_other_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = os.path.join(temp_dir, ".env")
            with open(env_path, "w", encoding="utf-8") as env_file:
                env_file.write("OTHER_KEY=1\nZALO_OA_TOKEN=old-token\n")

            save_env_value(env_path, "ZALO_OA_TOKEN", "new-token")

            with open(env_path, "r", encoding="utf-8") as env_file:
                content = env_file.read()

        self.assertEqual(content, "OTHER_KEY=1\nZALO_OA_TOKEN=new-token\n")

    def test_save_env_value_writes_blank_override_when_clearing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = os.path.join(temp_dir, ".env")

            save_env_value(env_path, "ZALO_OA_TOKEN", "")

            with open(env_path, "r", encoding="utf-8") as env_file:
                content = env_file.read()

        self.assertEqual(content, "ZALO_OA_TOKEN=\n")

    def test_load_application_env_prefers_persistent_values_over_project_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_env_path = os.path.join(temp_dir, "project.env")
            persistent_env_path = os.path.join(temp_dir, "persistent.env")

            with open(project_env_path, "w", encoding="utf-8") as env_file:
                env_file.write("ZALO_OA_TOKEN=old-token\nZALO_USER_ID=old-user\n")
            with open(persistent_env_path, "w", encoding="utf-8") as env_file:
                env_file.write("ZALO_OA_TOKEN=new-token\nZALO_USER_ID=\n")

            for key in ("ZALO_OA_TOKEN", "ZALO_USER_ID"):
                os.environ.pop(key, None)

            load_application_env(
                project_env_path=project_env_path,
                persistent_env_path=persistent_env_path,
            )

            token_value = os.getenv("ZALO_OA_TOKEN")
            user_id_value = os.getenv("ZALO_USER_ID")

        self.assertEqual(token_value, "new-token")
        self.assertEqual(user_id_value, "")


if __name__ == "__main__":
    unittest.main()
