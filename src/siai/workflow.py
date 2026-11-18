"""End-to-end, trainer-independent model improvement workflow."""

from dataclasses import asdict, dataclass
from collections.abc import Mapping
import json
import math
from pathlib import Path

from .core import Candidate, canonical
from .proposer import ModelCard, Observation, ProposalRequest


@dataclass(frozen=True)
class ImprovementResult:
    """Final state of a completed improvement run."""

    best: Candidate
    score: float
    evaluations: int
    trials: int
    journal: str


class ImprovementLoop:
    """Connect training feedback to proposals under a fixed evaluation budget.

    The trainer must expose ``train(config, *, seed, output_dir)`` and return an
    object with ``reward``, ``observations``, and ``checkpoint`` attributes. The
    proposer must expose ``propose(ProposalRequest)`` and return a ``Proposal``.
    Each configuration is trained from a fresh state for every paired seed.
    """

    def __init__(
        self,
        trainer,
        proposer,
        card: ModelCard,
        *,
        evaluation_budget: int = 9,
        seeds: tuple[int, ...] = (0, 1, 2),
        min_gain: float = 0.0,
        run_dir: str | Path = "siai-run",
    ):
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be nonempty and unique")
        if evaluation_budget < len(seeds):
            raise ValueError("evaluation budget must cover the baseline")
        if not math.isfinite(min_gain) or min_gain < 0:
            raise ValueError("minimum gain must be finite and nonnegative")
        self.trainer = trainer
        self.proposer = proposer
        self.card = card
        self.evaluation_budget = evaluation_budget
        self.seeds = tuple(seeds)
        self.min_gain = min_gain
        self.run_dir = Path(run_dir)

    def run(self, baseline: Candidate) -> ImprovementResult:
        """Run until the next paired-seed trial would exceed the budget."""

        self.run_dir.mkdir(parents=True, exist_ok=False)
        journal_path = self.run_dir / "journal.jsonl"
        incumbent = self._copy_candidate(baseline)
        seen = {incumbent.id}
        history: list[dict] = []
        evaluations = 0
        trials = 0

        with journal_path.open("x", encoding="utf-8") as journal:
            baseline_eval = self._evaluate(incumbent, trials)
            evaluations += baseline_eval["evaluations"]
            trials += 1
            baseline_row = self._trial_row(
                incumbent,
                parent=None,
                proposal=None,
                evaluation=baseline_eval,
                accepted=baseline_eval["mean"] is not None,
                evaluations=evaluations,
            )
            self._write(journal, baseline_row)
            history.append(baseline_row)
            if baseline_eval["mean"] is None:
                raise RuntimeError("baseline training failed")

            incumbent_score = baseline_eval["mean"]
            incumbent_observations = baseline_eval["observations"]
            feedback = incumbent_observations

            while evaluations + len(self.seeds) <= self.evaluation_budget:
                request = ProposalRequest(
                    card=self.card,
                    incumbent=self._copy_candidate(incumbent),
                    observations=tuple(feedback),
                    history=tuple(json.loads(canonical(history))),
                    remaining_evaluations=self.evaluation_budget - evaluations,
                )
                try:
                    proposal = self.proposer.propose(request)
                    candidate = proposal.candidate(incumbent)
                except Exception as exc:
                    self._write(
                        journal,
                        {
                            "type": "proposer_error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "evaluations": evaluations,
                        },
                    )
                    raise RuntimeError("proposer failed") from exc

                candidate = self._copy_candidate(candidate)
                if candidate.id in seen:
                    self._write(
                        journal,
                        {
                            "type": "duplicate",
                            "candidate": candidate.id,
                            "evaluations": evaluations,
                        },
                    )
                    break
                seen.add(candidate.id)

                candidate_eval = self._evaluate(candidate, trials)
                evaluations += candidate_eval["evaluations"]
                trials += 1
                candidate_score = candidate_eval["mean"]
                accepted = (
                    candidate_score is not None
                    and candidate_eval["guardrails_passed"]
                    and self._improved(candidate_score, incumbent_score)
                )
                row = self._trial_row(
                    candidate,
                    parent=incumbent.id,
                    proposal=asdict(proposal),
                    evaluation=candidate_eval,
                    accepted=accepted,
                    evaluations=evaluations,
                )
                self._write(journal, row)
                history.append(row)

                if accepted:
                    incumbent = candidate
                    incumbent_score = candidate_score
                    incumbent_observations = candidate_eval["observations"]
                    feedback = incumbent_observations
                else:
                    # Preserve incumbent evidence while exposing the latest failed
                    # attempt to the next proposal.
                    feedback = incumbent_observations + candidate_eval["observations"]

        return ImprovementResult(
            best=incumbent,
            score=incumbent_score,
            evaluations=evaluations,
            trials=trials,
            journal=str(journal_path),
        )

    def _evaluate(self, candidate: Candidate, trial: int) -> dict:
        scores: list[float] = []
        observations: list[Observation] = []
        checkpoints: list[str] = []
        errors: list[str] = []
        metric_samples: dict[str, list[float]] = {}
        evaluations = 0

        for seed in self.seeds:
            output_dir = self.run_dir / f"trial-{trial:03d}" / f"seed-{seed}"
            evaluations += 1
            try:
                result = self.trainer.train(
                    json.loads(canonical(candidate.config)),
                    seed=seed,
                    output_dir=output_dir,
                )
                score = float(result.reward)
                if not math.isfinite(score):
                    raise ValueError("non-finite model-card reward")
                raw_metrics = getattr(result, "metrics", {})
                if not isinstance(raw_metrics, Mapping):
                    raise ValueError("training metrics must be a mapping")
                metrics = {}
                for name, raw_value in raw_metrics.items():
                    if not isinstance(name, str) or not name:
                        raise ValueError("metric names must be nonempty strings")
                    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                        raise ValueError(f"invalid metric {name}")
                    value = float(raw_value)
                    if not math.isfinite(value):
                        raise ValueError(f"invalid metric {name}")
                    metrics[name] = value
                missing = [item.metric for item in self.card.guardrails if item.metric not in metrics]
                if missing:
                    raise ValueError(f"missing guardrail metrics: {missing}")
                scores.append(score)
                checkpoints.append(str(result.checkpoint))
                for name, value in metrics.items():
                    metric_samples.setdefault(name, []).append(value)
                    observations.append(
                        Observation(
                            id=f"trial-{trial}-seed-{seed}-metric-{name}",
                            kind="metric",
                            summary=f"Model-card search metric {name}",
                            value=value,
                        )
                    )
                for index, observation in enumerate(result.observations):
                    observations.append(
                        Observation(
                            id=f"trial-{trial}-seed-{seed}-{index}-{observation.id}",
                            kind=observation.kind,
                            summary=observation.summary,
                            value=observation.value,
                            artifact=observation.artifact,
                        )
                    )
            except Exception as exc:
                message = f"seed {seed}: {type(exc).__name__}: {exc}"
                errors.append(message)
                observations.append(
                    Observation(
                        id=f"trial-{trial}-seed-{seed}-error",
                        kind="failure",
                        summary=message,
                    )
                )
                break

        mean = (
            sum(scores) / len(scores)
            if not errors and len(scores) == len(self.seeds)
            else None
        )
        metric_means = {
            name: sum(values) / len(values)
            for name, values in metric_samples.items()
            if values
        }
        guardrails = []
        for item in self.card.guardrails:
            value = metric_means.get(item.metric)
            passed = value is not None and item.passes(value)
            guardrails.append({
                "metric": item.metric,
                "direction": item.direction,
                "threshold": item.threshold,
                "value": value,
                "passed": passed,
            })
        violations = [
            f"{item['metric']}={item['value']} must be {item['direction']} {item['threshold']}"
            for item in guardrails
            if not item["passed"]
        ]
        return {
            "scores": scores,
            "mean": mean,
            "metrics": metric_means,
            "guardrails": guardrails,
            "guardrails_passed": not errors and not violations,
            "guardrail_violations": violations,
            "observations": observations,
            "checkpoints": checkpoints,
            "errors": errors,
            "evaluations": evaluations,
        }

    def _improved(self, candidate: float, incumbent: float) -> bool:
        if self.card.direction == "maximize":
            return candidate > incumbent + self.min_gain
        return candidate < incumbent - self.min_gain

    @staticmethod
    def _copy_candidate(candidate: Candidate) -> Candidate:
        return Candidate(json.loads(canonical(candidate.config)), candidate.hypothesis)

    @staticmethod
    def _write(journal, row: dict) -> None:
        journal.write(canonical(row) + "\n")
        journal.flush()

    @staticmethod
    def _trial_row(candidate, *, parent, proposal, evaluation, accepted, evaluations):
        return {
            "type": "trial",
            "candidate": candidate.id,
            "parent": parent,
            "config": candidate.config,
            "hypothesis": candidate.hypothesis,
            "proposal": proposal,
            "scores": evaluation["scores"],
            "mean": evaluation["mean"],
            "metrics": evaluation["metrics"],
            "guardrails": evaluation["guardrails"],
            "guardrails_passed": evaluation["guardrails_passed"],
            "guardrail_violations": evaluation["guardrail_violations"],
            "observations": [asdict(item) for item in evaluation["observations"]],
            "checkpoints": evaluation["checkpoints"],
            "errors": evaluation["errors"],
            "accepted": accepted,
            "evaluations": evaluations,
        }
