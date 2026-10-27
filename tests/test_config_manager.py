"""Public-source API configuration never contains a usable default key."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config_manager


class ConfigManagerTests(unittest.TestCase):
    def test_no_bundled_key_and_environment_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(config_manager, "_CONFIG_PATH", Path(folder) / "config.json"):
                with patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
                    self.assertEqual(config_manager.get_api_key(), "")
                with patch.dict("os.environ", {"GEMINI_API_KEY": "environment-test-key"}):
                    self.assertEqual(config_manager.get_api_key(), "environment-test-key")

    def test_saved_key_can_be_cleared_without_affecting_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            with patch.object(config_manager, "_CONFIG_PATH", path):
                with patch.dict("os.environ", {"GEMINI_API_KEY": "environment-test-key"}):
                    config_manager.save_api_key("saved-test-key")
                    self.assertEqual(config_manager.get_api_key(), "saved-test-key")
                    self.assertTrue(config_manager.is_using_custom_key())
                    config_manager.clear_saved_api_key()
                    self.assertEqual(config_manager.get_api_key(), "environment-test-key")
                    self.assertFalse(config_manager.is_using_custom_key())
                    self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
