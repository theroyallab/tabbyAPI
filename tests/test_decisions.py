import unittest
import math

import torch

from endpoints.OAI.types.decisions import (
    ChoiceQuestion,
    DecisionsRequest,
    ScoreQuestion,
    YesNoQuestion,
)
from endpoints.OAI.utils.decisions import (
    DecisionValidationError,
    label_distribution,
    question_labels,
    render_question_message,
    validate_decisions_request,
)


def base_request(questions):
    return {"input": "A support ticket.", "questions": questions}


def valid_request():
    return DecisionsRequest.model_validate(
        base_request(
            [
                {
                    "id": "team",
                    "type": "choice",
                    "question": "Which team?",
                    "options": [{"name": "billing"}, {"name": "technical"}],
                },
                {"id": "angry", "type": "score", "question": "How angry?", "levels": ["Calm", "Fuming"]},
                {"id": "urgent", "type": "yes_no", "question": "Urgent?"},
            ]
        )
    )


class RenderTests(unittest.TestCase):
    def test_choice_message_renders_labels_in_order(self):
        question = valid_request().questions[0]
        message = render_question_message("ticket", question)
        self.assertIn("ticket\n\nWhich team?\nA: billing\nB: technical\n", message)
        self.assertTrue(message.endswith("Answer with only the letter of your choice."))

    def test_score_message_uses_level_indices(self):
        question = valid_request().questions[1]
        message = render_question_message("ticket", question)
        self.assertIn("0: Calm\n1: Fuming\n", message)

    def test_yes_no_without_descriptions_has_no_label_lines(self):
        question = valid_request().questions[2]
        message = render_question_message("ticket", question)
        self.assertNotIn("yes:", message)
        self.assertTrue(message.endswith("Answer with yes or no, all lowercase."))

    def test_non_string_input_becomes_compact_json(self):
        message = render_question_message({"a": 1}, valid_request().questions[2])
        self.assertTrue(message.startswith('{"a":1}\n\n'))


class ValidationTests(unittest.TestCase):
    def test_valid_request_passes(self):
        validate_decisions_request(valid_request())

    def test_blank_input_rejected(self):
        with self.assertRaises(ValueError):
            DecisionsRequest.model_validate({"input": "  ", "questions": [
                {"id": "u", "type": "yes_no", "question": "q"}]})

    def test_repeated_question_id_rejected(self):
        request = DecisionsRequest.model_validate(base_request([
            {"id": "same", "type": "yes_no", "question": "q"},
            {"id": "same", "type": "yes_no", "question": "q"},
        ]))
        with self.assertRaises(DecisionValidationError):
            validate_decisions_request(request)

    def test_option_count_bounds(self):
        one_option = base_request([
            {"id": "c", "type": "choice", "question": "q",
             "options": [{"name": "only"}]}])
        with self.assertRaises(DecisionValidationError):
            validate_decisions_request(DecisionsRequest.model_validate(one_option))

        too_many = base_request([
            {"id": "c", "type": "choice", "question": "q",
             "options": [{"name": f"o{i}"} for i in range(27)]}])
        with self.assertRaises(DecisionValidationError):
            validate_decisions_request(DecisionsRequest.model_validate(too_many))

    def test_repeated_option_names_rejected(self):
        request = DecisionsRequest.model_validate(base_request([
            {"id": "c", "type": "choice", "question": "q",
             "options": [{"name": "same"}, {"name": "SAME"}]}]))
        with self.assertRaises(DecisionValidationError):
            validate_decisions_request(request)

    def test_level_count_bounds(self):
        request = DecisionsRequest.model_validate(base_request([
            {"id": "s", "type": "score", "question": "q", "levels": ["only"]}]))
        with self.assertRaises(DecisionValidationError):
            validate_decisions_request(request)


class LabelDistributionTests(unittest.TestCase):
    def test_softmax_over_labels_only(self):
        # logits: "yes"=3, "no"=1, everything else -10
        logits = torch.full((32000,), -10.0)
        logits[7] = 3.0
        logits[9] = 1.0
        probs, label_mass = label_distribution(logits, [7, 9], temperature=1.0)
        # probabilities: softmax over the labels; the full-vocabulary
        # normalizer cancels
        self.assertAlmostEqual(probs[0], math.exp(3) / (math.exp(3) + math.exp(1)), places=6)
        # label_mass: full-vocabulary probability of the labels
        normalizer = math.exp(3) + math.exp(1) + 31998 * math.exp(-10)
        self.assertAlmostEqual(label_mass, (math.exp(3) + math.exp(1)) / normalizer, places=4)

    def test_temperature_divides_logit_difference(self):
        logits = torch.tensor([2.0, 0.0] + [-10.0] * 100)
        p_hot, _ = label_distribution(logits, [0, 1], temperature=0.5)
        p_cold, _ = label_distribution(logits, [0, 1], temperature=2.0)
        self.assertGreater(p_hot[0], p_cold[0])
        self.assertAlmostEqual(p_hot[0] + p_hot[1], 1.0)
        self.assertAlmostEqual(p_cold[0] + p_cold[1], 1.0)

    def test_label_mass_ignores_temperature(self):
        logits = torch.tensor([2.0, 0.0] + [-10.0] * 100)
        _, mass_one = label_distribution(logits, [0, 1], temperature=1.0)
        _, mass_two = label_distribution(logits, [0, 1], temperature=2.0)
        self.assertAlmostEqual(mass_one, mass_two)


class QuestionLabelsTests(unittest.TestCase):
    def test_choice_labels_follow_option_count(self):
        labels, names = question_labels(
            ChoiceQuestion(id="c", question="q", options=[{"name": "a"}, {"name": "b"}])
        )
        self.assertEqual(labels, ["A", "B"])
        self.assertEqual(names, ["a", "b"])

    def test_score_labels_are_indices(self):
        labels, _ = question_labels(ScoreQuestion(id="s", question="q", levels=["a", "b", "c"]))
        self.assertEqual(labels, ["0", "1", "2"])

    def test_yes_no_labels(self):
        labels, _ = question_labels(YesNoQuestion(id="y", question="q"))
        self.assertEqual(labels, ["yes", "no"])


if __name__ == "__main__":
    unittest.main()
