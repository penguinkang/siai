"""Real TRL training through ImprovementLoop with an offline Responses fixture.

This validates the full package boundary without an API key or model download.
The injected provider response is deterministic, so this is an integration test,
not evidence that an intelligent proposer discovered the improvement.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast

from siai import (
    Candidate,
    ImprovementLoop,
    IntelligentProposer,
    MetricGuardrail,
    ModelCard,
    OpenAIResponsesGenerator,
)
from siai.training import SFTAdapter, sft_factory


class OfflineResponses:
    """Small Responses API fixture that selects a known-safe test lever."""

    def __init__(self):
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        payload = json.loads(request["input"].split("REQUEST:\n", 1)[1])
        evidence_id = payload["observations"][0]["id"]
        output = json.dumps(
            {
                "lever": "learning_rate",
                "value": 0.003,
                "hypothesis": "A larger update rate may learn the repeated token pattern faster",
                "evidence_ids": [evidence_id],
                "expected_effect": "Potentially lower validation loss within four steps",
                "verification": "Compare negative validation loss from the paired fresh run",
            }
        )
        return SimpleNamespace(
            id="resp_offline_fixture",
            status="completed",
            output_text=output,
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
        )


def tokenizer_and_model(path: Path):
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "[EOS]": 2,
        "red": 3,
        "blue": 4,
        "green": 5,
    }
    raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    torch.manual_seed(10)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocab),
            n_positions=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=2,
            eos_token_id=2,
            pad_token_id=1,
        )
    )
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return tokenizer


def final_reward(checkpoint: str, tokenizer, texts: list[str]) -> float:
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    batch = tokenizer(texts, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    with torch.no_grad():
        loss = model(**batch, labels=labels).loss
    return -float(loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="examples/micro/trl-workflow-result.json")
    args = parser.parse_args()
    torch.set_num_threads(1)

    training_text = "red blue green red blue green"
    train = Dataset.from_dict({"text": [training_text] * 16})
    search = Dataset.from_dict({"text": ["red blue green red blue"] * 2})
    final_texts = ["blue green red blue green", "green red blue green red"]

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model_path = root / "base-model"
        model_path.mkdir()
        tokenizer = tokenizer_and_model(model_path)
        factory = sft_factory(
            "trl",
            str(model_path),
            train,
            max_steps=4,
            max_length=16,
        )

        search_losses = {}

        def score_search(trainer):
            loss = float(trainer.evaluate(eval_dataset=search)["eval_loss"])
            search_losses[id(trainer)] = loss
            return -loss

        def metric_search(trainer):
            return {"search_loss": search_losses[id(trainer)]}

        trainer = SFTAdapter(factory, score_search, metric_scorer=metric_search)
        responses = OfflineResponses()
        generate = OpenAIResponsesGenerator(
            "offline-proposal-fixture",
            client=SimpleNamespace(responses=responses),
            max_output_tokens=500,
        )
        proposer = IntelligentProposer(generate)
        card = ModelCard(
            objective="Lower held-out token loss within four optimizer steps",
            metric="negative_eval_loss",
            direction="maximize",
            levers={"learning_rate": "float in (0, 0.01]"},
            constraints=("Keep model, data, steps, and evaluator fixed",),
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
        accepted_rows = [row for row in rows if row["accepted"]]
        selected = accepted_rows[-1]
        test_reward = final_reward(selected["checkpoints"][0], tokenizer, final_texts)
        result = {
            "project": "tiny-local-lm-trl-workflow",
            "trainer": "trl.SFTTrainer",
            "provider_transport": "offline injected Responses API fixture",
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "trl", "datasets", "openai")
            },
            "baseline_search_reward": rows[0]["mean"],
            "proposed_search_reward": rows[1]["mean"],
            "baseline_metrics": rows[0]["metrics"],
            "proposed_metrics": rows[1]["metrics"],
            "proposed_guardrails_passed": rows[1]["guardrails_passed"],
            "accepted": rows[1]["accepted"],
            "selected_config": run.best.config,
            "final_test_reward": test_reward,
            "evaluations": run.evaluations,
            "provider_request": {
                "strict_schema": responses.requests[0]["text"]["format"]["strict"],
                "store": responses.requests[0]["store"],
            },
            "provider_metadata": rows[1]["proposal"]["provider"],
            "seed": 0,
        }

    result_path = Path(args.result)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
