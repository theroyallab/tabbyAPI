import pathlib
import tempfile
import unittest
from unittest.mock import patch

import common.model  # noqa: F401 - resolve import cycle ordering
from common import model
from endpoints.core.types.model import ModelCard, ModelCardParameters, ModelList
from endpoints.core.utils.model import get_current_model_list, get_model_list


class ModelCardTests(unittest.TestCase):
    def test_openai_fields(self):
        card = ModelCard(id="my-model")
        self.assertEqual(card.object, "model")
        self.assertEqual(card.owned_by, "tabbyAPI")
        self.assertIsInstance(card.created, int)

    def test_anthropic_fields_are_derived(self):
        card = ModelCard(id="my-model")
        self.assertEqual(card.type, "model")
        self.assertEqual(card.display_name, "my-model")

    def test_created_at_is_iso_utc(self):
        card = ModelCard(id="my-model", created=0)
        self.assertEqual(card.created_at, "1970-01-01T00:00:00Z")

    def test_explicit_values_are_kept(self):
        card = ModelCard(
            id="my-model", display_name="Pretty Name", created_at="2020-01-01T00:00:00Z"
        )
        self.assertEqual(card.display_name, "Pretty Name")
        self.assertEqual(card.created_at, "2020-01-01T00:00:00Z")


class ModelCardTokenFieldTests(unittest.TestCase):
    def test_token_fields_follow_max_seq_len(self):
        card = ModelCard(id="my-model", parameters=ModelCardParameters(max_seq_len=262144))

        self.assertEqual(card.max_input_tokens, 262144)
        self.assertEqual(card.max_tokens, 262144)

    def test_token_fields_are_null_without_parameters(self):
        card = ModelCard(id="my-model")

        self.assertIsNone(card.max_input_tokens)
        self.assertIsNone(card.max_tokens)

    def test_token_fields_are_serialized(self):
        card = ModelCard(id="my-model", parameters=ModelCardParameters(max_seq_len=4096))
        dumped = card.model_dump()

        self.assertEqual(dumped["max_input_tokens"], 4096)
        self.assertEqual(dumped["max_tokens"], 4096)


class ModelListTests(unittest.TestCase):
    def test_empty_list(self):
        listing = ModelList()
        self.assertFalse(listing.has_more)
        self.assertIsNone(listing.first_id)
        self.assertIsNone(listing.last_id)

    def test_ids_track_appends(self):
        # Callers build the list empty and append afterwards, so the ids
        # cannot be captured at construction time
        listing = ModelList()
        listing.data.append(ModelCard(id="a"))
        listing.data.append(ModelCard(id="b"))

        self.assertEqual(listing.first_id, "a")
        self.assertEqual(listing.last_id, "b")

    def test_ids_are_serialized(self):
        listing = ModelList(data=[ModelCard(id="a")])
        dumped = listing.model_dump()

        self.assertEqual(dumped["first_id"], "a")
        self.assertEqual(dumped["last_id"], "a")
        self.assertFalse(dumped["has_more"])


class DummyContainer:
    """Stands in for a loaded model container."""

    def __init__(self, model_dir: pathlib.Path):
        self.model_dir = model_dir
        self.draft_model_dir = model_dir / "draft"

    def model_info(self):
        return ModelCard(
            id=self.model_dir.name,
            parameters=ModelCardParameters(
                max_seq_len=4096,
                prompt_template="chat_template",
                prompt_template_content="{{ messages }}",
            ),
        )


class ModelListParametersTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.model_dir = pathlib.Path(self.temp_dir.name)
        (self.model_dir / "loaded-model").mkdir()
        (self.model_dir / "other-model").mkdir()

        self.container = DummyContainer(self.model_dir / "loaded-model")

    def test_directory_listing_fills_the_loaded_model_only(self):
        with patch.object(model, "container", self.container):
            listing = get_model_list(self.model_dir)

        cards = {card.id: card for card in listing.data}

        self.assertEqual(cards["loaded-model"].parameters.max_seq_len, 4096)
        self.assertIsNone(cards["other-model"].parameters)

    def test_listing_drops_the_template_content(self):
        # Every card would otherwise carry kilobytes of Jinja
        with patch.object(model, "container", self.container):
            listing = get_model_list(self.model_dir)

        card = next(card for card in listing.data if card.id == "loaded-model")

        self.assertEqual(card.parameters.prompt_template, "chat_template")
        self.assertIsNone(card.parameters.prompt_template_content)

    def test_directory_listing_without_a_loaded_model(self):
        with patch.object(model, "container", None):
            listing = get_model_list(self.model_dir)

        self.assertTrue(all(card.parameters is None for card in listing.data))

    async def test_current_model_list_fills_parameters(self):
        with patch.object(model, "container", self.container):
            listing = await get_current_model_list()

        self.assertEqual(listing.data[0].parameters.max_seq_len, 4096)

    async def test_draft_list_does_not_report_the_main_model(self):
        with patch.object(model, "container", self.container):
            listing = await get_current_model_list(model_type="draft")

        self.assertIsNone(listing.data[0].parameters)


if __name__ == "__main__":
    unittest.main()
