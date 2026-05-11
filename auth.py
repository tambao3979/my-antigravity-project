from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional


class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"


class Permission(str, Enum):
    VIEW_STREAMS = "view_streams"
    VIEW_STATUS = "view_status"
    VIEW_ALERTS = "view_alerts"
    VIEW_INFERENCE_LOGS = "view_inference_logs"
    MANAGE_CAMERAS = "manage_cameras"
    MANAGE_CONFIG = "manage_config"


ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.USER: {
        Permission.VIEW_STREAMS,
        Permission.VIEW_STATUS,
        Permission.VIEW_ALERTS,
        Permission.VIEW_INFERENCE_LOGS,
    },
    Role.ADMIN: set(Permission),
}


@dataclass(frozen=True)
class AuthenticatedUser:
    username: str
    role: Role

    def can(self, permission: Permission) -> bool:
        return permission in ROLE_PERMISSIONS[self.role]


class AuthService:
    def __init__(self, users: Mapping[str, tuple[str, Role | str]]):
        self._users: dict[str, tuple[str, Role]] = {}
        for username, (password, role) in users.items():
            clean_username = str(username).strip()
            clean_password = str(password)
            if not clean_username or not clean_password:
                continue
            self._users[clean_username] = (clean_password, Role(role))

    @classmethod
    def from_config(cls, config) -> "AuthService":
        return cls(
            {
                getattr(config, "AUTH_ADMIN_USERNAME", "admin"): (
                    getattr(config, "AUTH_ADMIN_PASSWORD", ""),
                    Role.ADMIN,
                ),
                getattr(config, "AUTH_USER_USERNAME", "user"): (
                    getattr(config, "AUTH_USER_PASSWORD", ""),
                    Role.USER,
                ),
            }
        )

    def authenticate(self, username: str, password: str) -> Optional[AuthenticatedUser]:
        clean_username = str(username).strip()
        record = self._users.get(clean_username)
        if record is None:
            return None

        expected_password, role = record
        if str(password) != expected_password:
            return None

        return AuthenticatedUser(username=clean_username, role=role)
