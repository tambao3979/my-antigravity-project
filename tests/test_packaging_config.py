import os
import unittest


class PackagingConfigTests(unittest.TestCase):
    def test_pyinstaller_spec_does_not_bundle_env_file(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec_path = os.path.join(project_root, "camera_ai.spec")
        with open(spec_path, "r", encoding="utf-8") as spec_file:
            spec_text = spec_file.read()

        self.assertNotRegex(spec_text, r"\(['\"]\.env['\"]\s*,")

    def test_env_example_does_not_recommend_default_passwords(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_example_path = os.path.join(project_root, ".env.example")
        with open(env_example_path, "r", encoding="utf-8") as env_file:
            env_example = env_file.read()

        self.assertNotIn("AUTH_ADMIN_PASSWORD=admin", env_example)
        self.assertNotIn("AUTH_USER_PASSWORD=user", env_example)

    def test_readme_references_existing_entrypoint(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        readme_path = os.path.join(project_root, "README.md")
        with open(readme_path, "r", encoding="utf-8") as readme_file:
            readme = readme_file.read()

        self.assertNotIn("main.py", readme)
        self.assertIn("gui.py", readme)

    def test_pyinstaller_spec_includes_runtime_icon_and_dynamic_gpu_import(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec_path = os.path.join(project_root, "camera_ai.spec")
        with open(spec_path, "r", encoding="utf-8") as spec_file:
            spec_text = spec_file.read()

        self.assertIn("'pynvml'", spec_text)
        self.assertIn("('logo.ico', '.')", spec_text)
        self.assertIn("icon='logo.ico'", spec_text)


if __name__ == "__main__":
    unittest.main()
