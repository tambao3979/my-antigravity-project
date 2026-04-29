import unittest

from auth import AuthService, Permission, Role


class AuthServiceTests(unittest.TestCase):
    def test_authenticates_admin_and_user_roles(self):
        service = AuthService(
            {
                "admin": ("admin-pass", Role.ADMIN),
                "viewer": ("viewer-pass", Role.USER),
            }
        )

        admin = service.authenticate("admin", "admin-pass")
        viewer = service.authenticate("viewer", "viewer-pass")

        self.assertIsNotNone(admin)
        self.assertEqual(admin.role, Role.ADMIN)
        self.assertIsNotNone(viewer)
        self.assertEqual(viewer.role, Role.USER)

    def test_rejects_unknown_or_invalid_credentials(self):
        service = AuthService({"admin": ("admin-pass", Role.ADMIN)})

        self.assertIsNone(service.authenticate("admin", "wrong"))
        self.assertIsNone(service.authenticate("missing", "admin-pass"))

    def test_user_role_is_read_only_while_admin_can_manage_configuration(self):
        service = AuthService(
            {
                "admin": ("admin-pass", Role.ADMIN),
                "viewer": ("viewer-pass", Role.USER),
            }
        )

        admin = service.authenticate("admin", "admin-pass")
        viewer = service.authenticate("viewer", "viewer-pass")

        self.assertTrue(admin.can(Permission.VIEW_STREAMS))
        self.assertTrue(admin.can(Permission.VIEW_ALERTS))
        self.assertTrue(admin.can(Permission.VIEW_INFERENCE_LOGS))
        self.assertTrue(admin.can(Permission.MANAGE_CAMERAS))
        self.assertTrue(admin.can(Permission.MANAGE_CONFIG))

        self.assertTrue(viewer.can(Permission.VIEW_STREAMS))
        self.assertTrue(viewer.can(Permission.VIEW_ALERTS))
        self.assertTrue(viewer.can(Permission.VIEW_INFERENCE_LOGS))
        self.assertFalse(viewer.can(Permission.MANAGE_CAMERAS))
        self.assertFalse(viewer.can(Permission.MANAGE_CONFIG))


if __name__ == "__main__":
    unittest.main()
