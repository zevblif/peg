"""settings.yaml generation stays in sync with the pinned GraphRAG config."""
import tempfile
import unittest

import _bootstrap  # noqa: F401
from pathlib import Path

import yaml

from peg_bench import config, patch_settings


class SettingsTest(unittest.TestCase):
    def test_template_placeholders_are_filled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            patch_settings.patch(root, config.DATA_MAIN / "smoke.csv")
            text = (root / "settings.yaml").read_text(encoding="utf-8")
            for placeholder in ("__CHAT_", "__EMBED_"):
                self.assertNotIn(placeholder, text)
            parsed = yaml.safe_load(text)
            chat = parsed["completion_models"]["default_completion_model"]
            self.assertEqual(chat["model"], config.CHAT_MODEL)
            self.assertEqual(chat["api_key"], "${PEG_CHAT_API_KEY}")
            self.assertTrue((root / "prompts" / "extract_graph.txt").exists())
            self.assertTrue((root / "input" / "smoke.csv").exists())


if __name__ == "__main__":
    unittest.main()
