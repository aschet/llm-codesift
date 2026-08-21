"""Rewriting opencode's configuration must not damage it."""
import json
import tempfile
import unittest
from pathlib import Path

from codesift import opencode
from codesift.config import Config


class TestStripComments(unittest.TestCase):
    def test_line_comments_removed(self):
        self.assertNotIn("note", opencode.strip_comments('{\n  // note\n  "a": 1\n}'))

    def test_urls_survive(self):
        """The scheme separator must not be mistaken for a comment."""
        source = '{"$schema": "https://opencode.ai/config.json"}'
        self.assertEqual(json.loads(opencode.strip_comments(source))["$schema"],
                         "https://opencode.ai/config.json")

    def test_realistic_config_round_trips(self):
        source = ('{\n'
                  '  "$schema": "https://opencode.ai/config.json",\n'
                  '  // choose the model\n'
                  '  "model": "ollama/qwen3:14b",\n'
                  '  "provider": {"ollama": {"options": '
                  '{"baseURL": "http://127.0.0.1:11434/v1"}}}\n'
                  '}')
        data = json.loads(opencode.strip_comments(source))
        self.assertEqual(data["model"], "ollama/qwen3:14b")
        self.assertEqual(data["provider"]["ollama"]["options"]["baseURL"],
                         "http://127.0.0.1:11434/v1")


class TestSync(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.path = self.tmp / "opencode.jsonc"
        self.cfg = Config(host="http://localhost:11434", models=["a:1", "b:2"])

    def test_dry_run_writes_nothing(self):
        opencode.run(self.cfg, self.path, write=False)
        self.assertFalse(self.path.exists())

    def test_creates_a_configuration_when_absent(self):
        opencode.run(self.cfg, self.path, write=True)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        provider = data["provider"]["ollama"]
        self.assertEqual(sorted(provider["models"]), ["a:1", "b:2"])
        self.assertEqual(provider["options"]["baseURL"], "http://localhost:11434/v1")

    def test_unrelated_settings_are_preserved(self):
        self.path.write_text(json.dumps({
            "model": "ollama/keep-me",
            "plugin": ["something"],
            "provider": {"other": {"npm": "x"}},
        }), encoding="utf-8")
        opencode.run(self.cfg, self.path, write=True)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(data["model"], "ollama/keep-me")
        self.assertEqual(data["plugin"], ["something"])
        self.assertIn("other", data["provider"])

    def test_existing_file_is_backed_up(self):
        self.path.write_text('{"model": "ollama/old"}', encoding="utf-8")
        opencode.run(self.cfg, self.path, write=True)
        backup = Path(str(self.path) + ".bak")
        self.assertTrue(backup.exists())
        self.assertIn("ollama/old", backup.read_text(encoding="utf-8"))

    def test_stale_models_are_replaced(self):
        self.path.write_text(json.dumps({"provider": {"ollama": {
            "models": {"gone:1": {"name": "gone:1"}}}}}), encoding="utf-8")
        opencode.run(self.cfg, self.path, write=True)
        models = json.loads(self.path.read_text(encoding="utf-8"))["provider"]["ollama"]["models"]
        self.assertNotIn("gone:1", models)
        self.assertIn("a:1", models)

    def test_invalid_configuration_is_refused(self):
        self.path.write_text("{ this is not json", encoding="utf-8")
        self.assertEqual(opencode.run(self.cfg, self.path, write=True), 1)
        self.assertIn("this is not json", self.path.read_text(encoding="utf-8"))

    def test_remote_host_is_used_in_the_base_url(self):
        cfg = Config(host="http://192.168.1.50:11434", models=["a:1"])
        opencode.run(cfg, self.path, write=True)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(data["provider"]["ollama"]["options"]["baseURL"],
                         "http://192.168.1.50:11434/v1")


if __name__ == "__main__":
    unittest.main()
