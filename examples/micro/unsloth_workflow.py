"""Real Unsloth MLX training through the shared SIAI workflow.

This example is intentionally kept on the non-release micro-project branch. It
uses a deterministic Responses API fixture, so it validates provider transport
and training integration rather than proposer intelligence.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile

from datasets import Dataset
import mlx.core as mx
import mlx.nn as nn
from unsloth import FastLanguageModel

from siai import (
    Candidate,
    ImprovementLoop,
    IntelligentProposer,
    MetricGuardrail,
    ModelCard,
    OpenAIResponsesGenerator,
)
from siai.training import SFTAdapter, sft_factory


MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


class OfflineResponses:
    def __init__(self):
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        payload = json.loads(request["input"].split("REQUEST:\n", 1)[1])
        evidence_id = payload["observations"][0]["id"]
        return SimpleNamespace(
            id="resp_offline_fixture",
            status="completed",
            output_text=json.dumps(
                {
                    "lever": "learning_rate",
                    "value": 0.001,
                    "hypothesis": "A larger LoRA update may learn the repeated pattern faster",
                    "evidence_ids": [evidence_id],
                    "expected_effect": "Lower held-out token loss within three steps",
                    "verification": "Compare paired negative cross-entropy rewards",
                }
            ),
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
        )


def negative_loss(model, tokenizer, texts):
    losses = []
    model.eval()
    for text in texts:
        tokens = mx.array([tokenizer.encode(text)])
        logits = model(tokens)
        loss = nn.losses.cross_entropy(logits[:, :-1, :], tokens[:, 1:]).mean()
        mx.eval(loss)
        losses.append(float(loss.item()))
    return -sum(losses) / len(losses)


def checkpoint_reward(checkpoint, texts):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint,
        max_seq_length=32,
        load_in_4bit=False,
    )
    return negative_loss(model, tokenizer, texts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="examples/micro/unsloth-workflow-result.json")
    args = parser.parse_args()

    train = Dataset.from_dict({"text": ["red blue green red blue green"] * 8})
    search_texts = ["red blue green red blue", "blue green red blue green"]
    final_texts = ["green red blue green red", "blue red green blue red"]

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        factory = sft_factory("unsloth", MODEL, train, max_steps=3, max_length=32)

        search_losses = {}

        def score_search(trainer):
            loss = -negative_loss(trainer.model, trainer.tokenizer, search_texts)
            search_losses[id(trainer)] = loss
            return -loss

        def metric_search(trainer):
            return {"search_loss": search_losses[id(trainer)]}

        trainer = SFTAdapter(factory, score_search, metric_scorer=metric_search)
        responses = OfflineResponses()
        proposer = IntelligentProposer(
            OpenAIResponsesGenerator(
                "offline-proposal-fixture",
                client=SimpleNamespace(responses=responses),
                max_output_tokens=500,
            )
        )
        card = ModelCard(
            objective="Lower held-out token loss within three optimizer steps",
            metric="negative_cross_entropy",
            direction="maximize",
            levers={"learning_rate": "float in (0, 0.01]"},
            constraints=("Keep model, LoRA config, data, steps, and evaluator fixed",),
            guardrails=(MetricGuardrail("search_loss", "at_most", 10),),
        )
        run = ImprovementLoop(
            trainer,
            proposer,
            card,
            evaluation_budget=2,
            seeds=(0,),
            run_dir=root / "run",
        ).run(Candidate({"learning_rate": 0.00001}))

        rows = [json.loads(line) for line in Path(run.journal).read_text().splitlines()]
        selected = [row for row in rows if row["accepted"]][-1]
        result = {
            "project": "smollm2-unsloth-mlx-workflow",
            "trainer": "unsloth.UnslothTrainer",
            "backend": "MLX",
            "model": MODEL,
            "provider_transport": "offline injected Responses API fixture",
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("unsloth", "mlx", "transformers", "trl", "datasets")
            },
            "baseline_search_reward": rows[0]["mean"],
            "proposed_search_reward": rows[1]["mean"],
            "baseline_metrics": rows[0]["metrics"],
            "proposed_metrics": rows[1]["metrics"],
            "proposed_guardrails_passed": rows[1]["guardrails_passed"],
            "accepted": rows[1]["accepted"],
            "selected_config": run.best.config,
            "final_test_reward": checkpoint_reward(selected["checkpoints"][0], final_texts),
            "evaluations": run.evaluations,
            "provider_request": {
                "strict_schema": responses.requests[0]["text"]["format"]["strict"],
                "store": responses.requests[0]["store"],
            },
            "provider_metadata": rows[1]["proposal"]["provider"],
            "seed": 0,
            "compatibility_note": (
                "The MLX trainer has no public evaluate method; the injected scorer computes "
                "cross-entropy directly from trainer.model and trainer.tokenizer."
            ),
        }

    result_path = Path(args.result)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
