import pathlib
import tempfile
import unittest

from endpoints.core.utils.model import get_model_list, relative_model_id


def make_model(directory: pathlib.Path):
    """Create the smallest thing that counts as a checkpoint directory."""

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    return directory


def listed_ids(model_root: pathlib.Path, draft_model_path=None):
    return [card.id for card in get_model_list(model_root, draft_model_path).data]


class GetModelListTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def test_flat_layout_lists_directory_names(self):
        # Layouts that already work must keep producing the exact same ids.
        make_model(self.root / "flat-model-a")
        make_model(self.root / "flat-model-b")

        self.assertEqual(
            listed_ids(self.root),
            ["flat-model-a", "flat-model-b"],
        )

    def test_nested_quant_layout_lists_relative_path(self):
        # The bug: this model could be loaded but never advertised.
        make_model(self.root / "nested-model" / "exl3" / "4bpw")

        self.assertEqual(listed_ids(self.root), ["nested-model/exl3/4bpw"])

    def test_parent_folder_is_not_listed_as_a_model(self):
        # A parent with no checkpoint of its own is only a container.
        make_model(self.root / "nested-model" / "exl3" / "4bpw")

        self.assertNotIn("nested-model", listed_ids(self.root))

    def test_huggingface_cache_is_not_listed(self):
        # `.cache` holds snapshot config.json files, so a plain recursive
        # search would advertise them as models.
        make_model(self.root / ".cache" / "huggingface" / "xlm-roberta-large")
        make_model(self.root / "real-model")

        self.assertEqual(listed_ids(self.root), ["real-model"])

    def test_checkpoint_subfolders_are_not_listed(self):
        # A checkpoint wins: its own subfolders are not searched deeper.
        make_model(self.root / "model-with-snapshots")
        make_model(self.root / "model-with-snapshots" / "snapshots" / "abc123")

        self.assertEqual(listed_ids(self.root), ["model-with-snapshots"])

    def test_draft_model_tree_is_excluded(self):
        draft = make_model(self.root / "draft" / "nested")

        self.assertEqual(listed_ids(self.root, str(draft)), [])

    def test_empty_folder_is_not_listed(self):
        (self.root / "not-a-model").mkdir()

        self.assertEqual(listed_ids(self.root), [])

    def test_ids_are_loadable_paths(self):
        # Whatever is advertised must resolve on disk, because that resolved
        # path is what /v1/model/load builds from the id.
        make_model(self.root / "nested-model" / "exl3" / "4bpw")

        for model_id in listed_ids(self.root):
            self.assertTrue((self.root / model_id).is_dir(), model_id)


class RelativeModelIdTests(unittest.TestCase):
    def test_nested_path_becomes_a_posix_relative_path(self):
        root = pathlib.Path("/models")
        self.assertEqual(
            relative_model_id(root / "a" / "b", root),
            "a/b",
        )

    def test_model_outside_model_dir_falls_back_to_name(self):
        # A model loaded from an absolute path outside model_dir has no
        # relative form, so the old id shape is kept.
        self.assertEqual(
            relative_model_id(pathlib.Path("/elsewhere/my-model"), pathlib.Path("/models")),
            "my-model",
        )


if __name__ == "__main__":
    unittest.main()
