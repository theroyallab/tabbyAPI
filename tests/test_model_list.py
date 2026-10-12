import asyncio
import json
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from common import model
from common.tabby_config import config
from endpoints.core.utils.model import (
    get_current_model_list,
    get_model_list,
    relative_model_id,
)


def make_model(directory: pathlib.Path, model_config=None):
    """Create the smallest thing that counts as a checkpoint directory."""

    directory.mkdir(parents=True, exist_ok=True)
    if model_config is None:
        model_config = {"architectures": ["LlamaForCausalLM"]}
    (directory / "config.json").write_text(json.dumps(model_config), encoding="utf-8")
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

    def test_draft_dir_shared_with_model_dir_hides_nothing(self):
        # A common setup: drafts live alongside the other models
        make_model(self.root / "big" / "exl3" / "4bpw")
        make_model(self.root / "small")

        self.assertEqual(listed_ids(self.root, str(self.root)), ["big/exl3/4bpw", "small"])
        self.assertEqual(listed_ids(self.root, str(self.root.parent)), ["big/exl3/4bpw", "small"])

    def test_empty_folder_is_not_listed(self):
        (self.root / "not-a-model").mkdir()

        self.assertEqual(listed_ids(self.root), [])

    def test_config_must_name_an_architecture(self):
        # config.json alone doesn't make a model: other tools leave them too
        valid = [
            {"architectures": "LlamaForCausalLM"},
            {"architectures": ["LlamaForCausalLM"]},
            {"architectures": ["A", "B"]},
        ]
        invalid = [
            {},
            {"architectures": []},
            {"architectures": ""},
            {"architectures": [1]},
            {"architectures": ["A", None]},
            {"architectures": {"name": "A"}},
            ["architectures"],
        ]
        for index, cfg in enumerate(valid):
            make_model(self.root / f"valid-{index}", cfg)
        for index, cfg in enumerate(invalid):
            make_model(self.root / f"invalid-{index}", cfg)
        broken = self.root / "broken"
        broken.mkdir()
        (broken / "config.json").write_text("{not json", encoding="utf-8")

        self.assertEqual(listed_ids(self.root), ["valid-0", "valid-1", "valid-2"])

    def test_folder_with_non_model_config_is_searched(self):
        # Not a checkpoint, so models inside it are still found
        make_model(self.root / "collection", {"name": "my quants"})
        make_model(self.root / "collection" / "4bpw")

        self.assertEqual(listed_ids(self.root), ["collection/4bpw"])

    def test_ids_are_loadable_paths(self):
        # Whatever is advertised must resolve on disk, because that resolved
        # path is what /v1/model/load builds from the id.
        make_model(self.root / "nested-model" / "exl3" / "4bpw")

        for model_id in listed_ids(self.root):
            self.assertTrue((self.root / model_id).is_dir(), model_id)


class CurrentModelListTests(unittest.TestCase):
    """The loaded model's id is relative to the directory its kind is listed from."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def current_id(self, model_type):
        listing = asyncio.run(get_current_model_list(model_type))
        return [card.id for card in listing.data]

    def test_each_kind_uses_its_own_directory(self):
        models = self.root / "models"
        drafts = self.root / "drafts"
        embeddings = self.root / "embeddings"
        container = SimpleNamespace(
            model_dir=make_model(models / "big" / "exl3" / "4bpw"),
            draft_model_dir=make_model(drafts / "small" / "exl3" / "8bpw"),
            max_seq_len=4096,
        )
        embeddings_container = SimpleNamespace(model_dir=make_model(embeddings / "bge" / "v1"))

        with (
            mock.patch.object(model, "container", container),
            mock.patch.object(model, "embeddings_container", embeddings_container),
            mock.patch.object(config.model, "model_dir", str(models)),
            mock.patch.object(config.draft_model, "draft_model_dir", str(drafts)),
            mock.patch.object(config.embeddings, "embedding_model_dir", str(embeddings)),
        ):
            self.assertEqual(self.current_id("model"), ["big/exl3/4bpw"])
            self.assertEqual(self.current_id("draft"), ["small/exl3/8bpw"])
            self.assertEqual(self.current_id("embedding"), ["bge/v1"])


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
