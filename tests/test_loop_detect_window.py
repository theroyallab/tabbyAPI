import unittest

from common.sampling import BaseSamplerRequest
from endpoints.OAI.utils.chat_completion import _compose_serialize_stream_chunk
from endpoints.OAI.utils.completion import _compose_response


class LoopDetectWindowTests(unittest.TestCase):
    def test_default_loop_detect_window_maps_to_exllamav3_stop_on_loop(self):
        request = BaseSamplerRequest()

        self.assertEqual(request.loop_detect_window, 800)
        self.assertEqual(request.get_stop_on_loop(), (800, 2))

    def test_loop_detect_window_can_be_disabled(self):
        request = BaseSamplerRequest(loop_detect_window=0)

        self.assertIsNone(request.get_stop_on_loop())

    def test_completion_response_exposes_loop_detected_eos_reason(self):
        response = _compose_response(
            "request-id",
            [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "eos_reason": "loop_detected",
                    "content": "",
                }
            ],
            "model",
            False,
        )

        self.assertEqual(response.choices[0].finish_reason, "stop")
        self.assertEqual(response.choices[0].eos_reason, "loop_detected")

    def test_stream_chunk_exposes_loop_detected_eos_reason(self):
        chunk, data, finish_reason, is_empty = _compose_serialize_stream_chunk(
            "request-id",
            {
                "index": 0,
                "finish_reason": "stop",
                "eos_reason": "loop_detected",
            },
        )

        self.assertEqual(finish_reason, "stop")
        self.assertFalse(is_empty)
        self.assertIn('"eos_reason": "loop_detected"', chunk)
        self.assertEqual(data["choices"][0]["eos_reason"], "loop_detected")


if __name__ == "__main__":
    unittest.main()
