import os
import sys

from dotenv import load_dotenv


APP_NAME = "CameraAI"


def get_project_env_path(project_root: str | None = None) -> str:
    if project_root:
        root_dir = project_root
    elif getattr(sys, "frozen", False):
        root_dir = os.path.dirname(os.path.abspath(sys.executable))
    else:
        root_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root_dir, ".env")


def get_persistent_app_dir(app_name: str = APP_NAME, appdata_root: str | None = None) -> str:
    base_dir = appdata_root or os.getenv("APPDATA")
    if base_dir:
        return os.path.join(base_dir, app_name)
    return os.path.join(os.path.expanduser("~"), f".{app_name.lower()}")


def get_persistent_env_path(app_name: str = APP_NAME, appdata_root: str | None = None) -> str:
    return os.path.join(get_persistent_app_dir(app_name=app_name, appdata_root=appdata_root), ".env")


def save_env_value(env_path: str, key: str, value: str) -> None:
    os.makedirs(os.path.dirname(env_path), exist_ok=True)

    lines: list[str] = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as env_file:
            lines = env_file.readlines()

    updated_lines: list[str] = []
    found = False
    for line in lines:
        if line.startswith(f"{key}="):
            updated_lines.append(f"{key}={value}\n")
            found = True
        else:
            updated_lines.append(line)

    if not found:
        updated_lines.append(f"{key}={value}\n")

    with open(env_path, "w", encoding="utf-8") as env_file:
        env_file.writelines(updated_lines)


def load_application_env(
    project_env_path: str | None = None,
    persistent_env_path: str | None = None,
) -> tuple[str, str]:
    resolved_project_env_path = project_env_path or get_project_env_path()
    resolved_persistent_env_path = persistent_env_path or get_persistent_env_path()

    if os.path.exists(resolved_project_env_path):
        load_dotenv(resolved_project_env_path, override=False)
    if os.path.exists(resolved_persistent_env_path):
        load_dotenv(resolved_persistent_env_path, override=True)

    return resolved_project_env_path, resolved_persistent_env_path
