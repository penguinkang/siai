"""Small, provider-independent experiment kernel. Callbacks are trusted code."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


@dataclass(frozen=True)
class Candidate:
    config: dict
    hypothesis: str = "baseline"

    @property
    def id(self):
        return hashlib.sha256(canonical(self.config).encode()).hexdigest()


@dataclass(frozen=True)
class Result:
    best: Candidate
    score: float
    evaluations: int


class Experiment:
    """Maximize mean score over fixed paired seeds within an evaluation budget.

    Evaluator signature: (config, seed, split) -> finite float.
    Proposer signature: (incumbent, immutable history tuple) -> Candidate.
    Budget includes the baseline. Each evaluation is one seed/candidate pair.
    A new journal is required for each run; resume is deliberately unsupported.
    """

    def __init__(self, evaluate: Callable, propose: Callable, *, budget=30,
                 seeds=(0, 1, 2), min_gain=0.0, journal="run.jsonl"):
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be nonempty and unique")
        if budget < len(seeds) or not math.isfinite(min_gain) or min_gain < 0:
            raise ValueError("invalid budget or minimum gain")
        self.evaluate, self.propose = evaluate, propose
        self.budget, self.seeds, self.min_gain = budget, tuple(seeds), min_gain
        self.journal = Path(journal)

    def run(self, baseline: Candidate) -> Result:
        history, seen, used = [], set(), 0
        best = Candidate(json.loads(canonical(baseline.config)), baseline.hypothesis)
        best_score = -math.inf
        # Exclusive creation prevents accidental overwrite of prior evidence.
        with self.journal.open("x", encoding="utf-8") as log:
            while used + len(self.seeds) <= self.budget:
                candidate = best if not history else self.propose(
                    Candidate(json.loads(canonical(best.config)), best.hypothesis),
                    tuple(json.loads(canonical(history))))
                candidate = Candidate(json.loads(canonical(candidate.config)), candidate.hypothesis)
                if candidate.id in seen:
                    break
                seen.add(candidate.id)
                parent = best.id if history else None
                scores, error = [], None
                for seed in self.seeds:
                    used += 1
                    try:
                        score = float(self.evaluate(json.loads(canonical(candidate.config)), seed, "search"))
                        if not math.isfinite(score):
                            raise ValueError("non-finite score")
                        scores.append(score)
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        break
                mean = sum(scores) / len(scores) if not error else None
                accepted = mean is not None and mean > best_score + self.min_gain
                row = dict(candidate=candidate.id, parent=parent, config=candidate.config,
                           hypothesis=candidate.hypothesis, seeds=list(self.seeds), scores=scores,
                           mean=mean, accepted=accepted, error=error, evaluations=used)
                log.write(canonical(row) + "\n")
                log.flush()
                history.append(row)
                if accepted:
                    best, best_score = candidate, mean
                if len(history) == 1 and error:
                    raise RuntimeError("baseline evaluation failed: " + error)
        return Result(best, best_score, used)
