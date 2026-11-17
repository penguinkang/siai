import json
import tempfile
import unittest
from pathlib import Path

from siai import (
    Candidate,
    ImprovementLoop,
    ModelCard,
    Observation,
    Proposal,
)
from siai.training import TrainingResult


class FakeTrainer:
    def __init__(self, fail_value=None):
        self.fail_value = fail_value
        self.calls = []

    def train(self, config, *, seed, output_dir):
        self.calls.append((dict(config), seed, Path(output_dir)))
        if config["value"] == self.fail_value:
            raise RuntimeError("training exploded")
        path = Path(output_dir)
        path.mkdir(parents=True)
        checkpoint = path / "checkpoint"
        checkpoint.mkdir()
        return TrainingResult(
            reward=float(config["value"]),
            observations=(
                Observation("reward", "task", "Measured search reward", config["value"]),
            ),
            checkpoint=str(checkpoint),
        )


class ScriptedProposer:
    def __init__(self, values):
        self.values = iter(values)
        self.requests = []

    def propose(self, request):
        self.requests.append(request)
        value = next(self.values)
        return Proposal(
            lever="value",
            value=value,
            hypothesis=f"Try value {value}",
            evidence_ids=(request.observations[0].id,),
            expected_effect="Potential reward improvement",
            verification="Compare paired seeds",
        )


class ImprovementLoopTests(unittest.TestCase):
    def card(self, direction="maximize"):
        return ModelCard("Improve reward", "score", direction, {"value": "finite number"})

    def test_end_to_end_accepts_and_rejects_with_complete_journal(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer = FakeTrainer()
            proposer = ScriptedProposer([2, 1])
            run_dir = Path(temp) / "run"
            result = ImprovementLoop(
                trainer,
                proposer,
                self.card(),
                evaluation_budget=6,
                seeds=(3, 7),
                run_dir=run_dir,
            ).run(Candidate({"value": 0}))

            self.assertEqual(result.best.config, {"value": 2})
            self.assertEqual(result.score, 2)
            self.assertEqual(result.evaluations, 6)
            self.assertEqual(result.trials, 3)
            self.assertEqual(len(trainer.calls), 6)
            self.assertEqual(len(proposer.requests), 2)
            self.assertEqual(proposer.requests[0].remaining_evaluations, 4)
            self.assertTrue(all(item.id.startswith("trial-") for item in proposer.requests[0].observations))

            rows = [json.loads(line) for line in Path(result.journal).read_text().splitlines()]
            self.assertEqual([row["accepted"] for row in rows], [True, True, False])
            self.assertEqual(rows[1]["proposal"]["lever"], "value")
            self.assertEqual(rows[2]["parent"], rows[1]["candidate"])
            self.assertEqual(len(rows[0]["checkpoints"]), 2)

    def test_minimizing_metric_promotes_lower_score(self):
        with tempfile.TemporaryDirectory() as temp:
            result = ImprovementLoop(
                FakeTrainer(),
                ScriptedProposer([1]),
                self.card("minimize"),
                evaluation_budget=2,
                seeds=(0,),
                run_dir=Path(temp) / "run",
            ).run(Candidate({"value": 2}))
            self.assertEqual(result.best.config, {"value": 1})

    def test_failed_candidate_is_recorded_and_does_not_replace_incumbent(self):
        with tempfile.TemporaryDirectory() as temp:
            result = ImprovementLoop(
                FakeTrainer(fail_value=9),
                ScriptedProposer([9]),
                self.card(),
                evaluation_budget=4,
                seeds=(0, 1),
                run_dir=Path(temp) / "run",
            ).run(Candidate({"value": 2}))
            self.assertEqual(result.best.config, {"value": 2})
            self.assertEqual(result.evaluations, 3)
            rows = [json.loads(line) for line in Path(result.journal).read_text().splitlines()]
            self.assertFalse(rows[1]["accepted"])
            self.assertIn("training exploded", rows[1]["errors"][0])

    def test_baseline_failure_aborts_but_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            loop = ImprovementLoop(
                FakeTrainer(fail_value=0),
                ScriptedProposer([]),
                self.card(),
                seeds=(0,),
                run_dir=run_dir,
            )
            with self.assertRaisesRegex(RuntimeError, "baseline training failed"):
                loop.run(Candidate({"value": 0}))
            row = json.loads((run_dir / "journal.jsonl").read_text())
            self.assertFalse(row["accepted"])
            self.assertEqual(row["observations"][0]["kind"], "failure")


if __name__ == "__main__":
    unittest.main()
