# Core experiment API

Import these objects from `siai`.

## `Candidate`

```python
Candidate(config: dict, hypothesis: str = "baseline")
```

An immutable experiment candidate. `config` must be JSON-serializable. The `id`
property returns a SHA-256 fingerprint of the canonicalized configuration. The
hypothesis does not affect the fingerprint.

## `Result`

```python
Result(best: Candidate, score: float, evaluations: int)
```

The selected candidate, its mean search score, and the number of attempted
candidate/seed evaluations.

## `Experiment`

```python
Experiment(
    evaluate,
    propose,
    *,
    budget: int = 30,
    seeds: tuple = (0, 1, 2),
    min_gain: float = 0.0,
    journal: str | Path = "run.jsonl",
)
```

Maximizes mean score over fixed paired seeds within an evaluation budget.

### `evaluate`

```python
evaluate(config: dict, seed: int, split: str) -> float
```

Must return a finite float. `Experiment` passes `"search"` as the split. Return
negative loss for a minimized objective.

### `propose`

```python
propose(incumbent: Candidate, history: tuple[dict, ...]) -> Candidate
```

Receives defensive copies. Returning an already evaluated configuration stops
the run.

### `run`

```python
run(baseline: Candidate) -> Result
```

Creates the journal exclusively; an existing path raises `FileExistsError`.
Baseline failure raises `RuntimeError`. Later failures are recorded and rejected.
Resume and concurrent writers are not supported in v0.1.0.
