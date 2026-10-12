import pathlib
import unittest
from types import SimpleNamespace

from common import model
from common.tabby_config import config
from endpoints.OAI.utils.common_ import _is_loaded_model


class IsLoadedModelTests(unittest.TestCase):
    """
    Model identity checks must compare full paths: quant-style layouts give
    unrelated models the same directory basename (<model>/exl3/<bpw>), which
    used to make inline loading silently skip the model swap.
    """

    def setUp(self):
        self.original_container = model.container
        self.original_model_dir = config.model.model_dir
        config.model.model_dir = "/models"
        model.container = SimpleNamespace(
            model_dir=pathlib.Path("/models/model-a/exl3/4.00bpw"),
            loaded=True,
        )

    def tearDown(self):
        model.container = self.original_container
        config.model.model_dir = self.original_model_dir

    def test_advertised_basename_matches(self):
        self.assertTrue(_is_loaded_model("4.00bpw"))

    def test_full_path_of_loaded_model_matches(self):
        self.assertTrue(_is_loaded_model("/models/model-a/exl3/4.00bpw"))

    def test_relative_path_of_loaded_model_matches(self):
        self.assertTrue(_is_loaded_model("model-a/exl3/4.00bpw"))

    def test_same_basename_different_model_does_not_match(self):
        self.assertFalse(_is_loaded_model("/models/model-b/exl3/4.00bpw"))
        self.assertFalse(_is_loaded_model("model-b/exl3/4.00bpw"))

    def test_different_name_does_not_match(self):
        self.assertFalse(_is_loaded_model("other-model"))

    def test_not_loaded(self):
        model.container = SimpleNamespace(
            model_dir=pathlib.Path("/models/model-a/exl3/4.00bpw"),
            loaded=False,
        )
        self.assertFalse(_is_loaded_model("4.00bpw"))
        model.container = None
        self.assertFalse(_is_loaded_model("4.00bpw"))


if __name__ == "__main__":
    unittest.main()


class SymlinkAliasTests(unittest.TestCase):
    """A symlink alias for the loaded model directory is the loaded model (#379)."""

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.tmp.name)
        self.real = root / "ArtusDev_Qwen3-Coder-30B-EXL3"
        self.real.mkdir()
        (root / "coder").symlink_to(self.real, target_is_directory=True)
        (root / "other").mkdir()

        self.original_container = model.container
        self.original_model_dir = config.model.model_dir
        config.model.model_dir = str(root)
        model.container = SimpleNamespace(model_dir=self.real.resolve(), loaded=True)

    def tearDown(self):
        model.container = self.original_container
        config.model.model_dir = self.original_model_dir
        self.tmp.cleanup()

    def test_symlink_alias_is_the_loaded_model(self):
        self.assertTrue(_is_loaded_model("coder"))
        self.assertTrue(_is_loaded_model("ArtusDev_Qwen3-Coder-30B-EXL3"))

    def test_unrelated_directory_is_not(self):
        self.assertFalse(_is_loaded_model("other"))
        self.assertFalse(_is_loaded_model("missing"))
