import json
import unittest
from types import SimpleNamespace

from siai import OpenAIResponsesGenerator, ProviderError


class FakeResponses:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        return self.response


class ProviderTests(unittest.TestCase):
    def test_openai_responses_uses_strict_schema_and_records_metadata(self):
        output = json.dumps({
            "lever": "learning_rate",
            "value": 0.0005,
            "hypothesis": "Reduce instability",
            "evidence_ids": ["loss"],
            "expected_effect": "Potential reward gain",
            "verification": "Compare paired seeds",
        })
        responses = FakeResponses(SimpleNamespace(
            id="resp_123",
            status="completed",
            output_text=output,
            usage=SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=30),
        ))
        generator = OpenAIResponsesGenerator(
            "test-model",
            client=SimpleNamespace(responses=responses),
            reasoning_effort="low",
        )

        self.assertEqual(generator("Return a proposal"), output)
        request = responses.requests[0]
        self.assertFalse(request["store"])
        self.assertTrue(request["text"]["format"]["strict"])
        self.assertFalse(request["text"]["format"]["schema"]["additionalProperties"])
        self.assertEqual(request["reasoning"], {"effort": "low"})
        self.assertEqual(generator.last_metadata.response_id, "resp_123")
        self.assertEqual(generator.last_metadata.usage["total_tokens"], 30)

    def test_missing_output_raises_after_metadata_is_preserved(self):
        responses = FakeResponses(SimpleNamespace(
            id="resp_refusal", status="incomplete", output_text="", usage=None
        ))
        generator = OpenAIResponsesGenerator(
            "test-model", client=SimpleNamespace(responses=responses)
        )
        with self.assertRaises(ProviderError):
            generator("Return a proposal")
        self.assertEqual(generator.last_metadata.status, "incomplete")

    def test_configuration_validation(self):
        client = SimpleNamespace(responses=FakeResponses(None))
        for kwargs in (
            {"model": ""},
            {"model": "test", "timeout": 0},
            {"model": "test", "max_output_tokens": 0},
            {"model": "test", "reasoning_effort": "extreme"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OpenAIResponsesGenerator(client=client, **kwargs)


if __name__ == "__main__":
    unittest.main()
