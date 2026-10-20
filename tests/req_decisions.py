"""Live check for /v1/decisions against a running tabbyAPI server.

Run with the server's own venv, e.g.:
    ~/.venvs/qwen38-exl3/bin/python req_decisions.py [BASE_URL] [MODEL]
"""

import json
import sys

import httpx

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8081/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "gemma4-26b"

REQUEST = {
    "model": MODEL,
    "input": "I've been trying to connect my Stripe account for 3 days and the "
    "integration keeps failing. I'm losing sales.",
    "questions": [
        {
            "id": "team",
            "type": "choice",
            "question": "Which team should handle this ticket?",
            "options": [
                {"name": "billing", "description": "Payment or subscription issues"},
                {"name": "technical", "description": "Bugs or integration problems"},
                {"name": "sales", "description": "Pricing or account questions"},
            ],
        },
        {
            "id": "frustration",
            "type": "score",
            "question": "How frustrated is the customer?",
            "levels": ["Calm", "Frustrated but civil", "Very angry"],
        },
        {
            "id": "urgent",
            "type": "yes_no",
            "question": "The customer needs an answer today.",
        },
    ],
}


def main():
    response = httpx.post(f"{BASE_URL}/decisions", json=REQUEST, timeout=120)
    print(f"HTTP {response.status_code}")
    response.raise_for_status()
    data = response.json()

    print(f"model={data['model']} prompt_format_version={data['prompt_format_version']}")
    for question_id, answer in data["answers"].items():
        print(f"\n{question_id} ({answer['type']}):")
        print(f"  probabilities: {json.dumps(answer['probabilities'])}")
        print(f"  label_mass:    {answer['label_mass']:.4f}")
        if "choice" in answer:
            print(f"  choice:        {answer['choice']}")
        if "score" in answer:
            print(f"  score:         {answer['score']:.2f}")
    print(f"\nusage: {data['usage']}")

    # Sanity: probabilities sum to 1, every question id is answered, and
    # decision-type answers carry their decision field
    for answer in data["answers"].values():
        assert abs(sum(answer["probabilities"].values()) - 1.0) < 1e-4
    assert set(data["answers"]) == {q["id"] for q in REQUEST["questions"]}
    assert "choice" in data["answers"]["team"]
    assert "score" in data["answers"]["frustration"]
    print("\nSanity checks passed")


if __name__ == "__main__":
    main()
