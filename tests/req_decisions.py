"""Live check of the decision endpoints against a running TabbyAPI server.

Sends the OpenAI Decisions request vLLM's end-to-end test uses to /v1/decisions
and the System One example from llama.cpp's server README to /v1/systemone, and
checks the response shapes the same way.

    python tests/req_decisions.py [BASE_URL] [API_KEY]
"""

import json
import sys

import httpx

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000/v1"
API_KEY = sys.argv[2] if len(sys.argv) > 2 else None
HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}


def approx(a, b, tol=1e-4):
    return abs(a - b) < tol


def check_decisions():
    response = httpx.post(
        f"{BASE_URL}/decisions",
        headers=HEADERS,
        json={
            "input": "The package arrived with a broken screen.",
            "questions": [
                {"type": "predicate", "instructions": "Is the screen broken?"},
                {
                    "type": "choice",
                    "name": "department",
                    "instructions": "Which department should handle this?",
                    "choices": [{"value": "returns"}, {"value": "billing"}],
                },
                {
                    "type": "score",
                    "name": "damage",
                    "instructions": "Rate the damage.",
                    "levels": [{"label": "undamaged"}, {"label": "broken"}],
                },
            ],
        },
        timeout=300,
    )
    print(f"/v1/decisions HTTP {response.status_code}")
    response.raise_for_status()
    body = response.json()
    print(json.dumps(body, indent=2))

    assert set(body) == {"model", "answers", "usage"}
    predicate, choice_answer, score = body["answers"]
    assert predicate["type"] == "predicate" and predicate["name"] is None
    assert 0 <= predicate["probability"] <= 1
    assert choice_answer["type"] == "choice" and choice_answer["name"] == "department"
    assert [p["value"] for p in choice_answer["probabilities"]] == ["returns", "billing"]
    assert (
        choice_answer["choice"]
        == max(choice_answer["probabilities"], key=lambda p: p["probability"])["value"]
    )
    assert approx(sum(p["probability"] for p in choice_answer["probabilities"]), 1)
    assert score["type"] == "score" and score["name"] == "damage"
    assert [p["label"] for p in score["probabilities"]] == ["undamaged", "broken"]
    assert approx(
        score["score"], sum(p["value"] * p["probability"] for p in score["probabilities"])
    )
    usage = body["usage"]
    assert usage["output_tokens"] == 3
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]

    rejected = httpx.post(
        f"{BASE_URL}/decisions",
        headers=HEADERS,
        json={
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "https://example.com/i.png"}],
                }
            ],
            "questions": [{"type": "predicate", "instructions": "Is this red?"}],
        },
        timeout=60,
    )
    assert rejected.status_code in (400, 422), rejected.text
    assert "text input only" in rejected.text


def check_systemone():
    response = httpx.post(
        f"{BASE_URL}/systemone",
        headers=HEADERS,
        json={
            "state": "Customer message: I was charged twice for my order last week and nobody has replied.",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "criteria": {"billing": None, "shipping": None, "technical": None},
                },
                "angry": {"type": "noul", "instructions": "Is the customer angry?"},
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent is this?",
                    "criteria": ["can wait", "this week", "today", "right now"],
                },
            },
        },
        timeout=300,
    )
    print(f"/v1/systemone HTTP {response.status_code}")
    response.raise_for_status()
    body = response.json()
    print(json.dumps(body, indent=2))

    answers = body["answers"]
    assert set(answers) == {"route", "angry", "urgency"}
    assert answers["route"]["choice"] in {"billing", "shipping", "technical"}
    assert approx(sum(answers["route"]["probabilities"].values()), 1)
    assert 0 <= answers["angry"]["noul"] <= 1
    assert answers["urgency"]["legend"] == {
        "0": "can wait",
        "1": "this week",
        "2": "today",
        "3": "right now",
    }
    assert 0 <= answers["urgency"]["score"] <= 3
    assert body["usage"]["output_tokens"] == 3
    for diag in body["diagnostics"].values():
        assert 0 <= diag["label_mass"] <= 1


if __name__ == "__main__":
    check_decisions()
    check_systemone()
    print("\nSanity checks passed")
