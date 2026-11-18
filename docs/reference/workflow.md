# Improvement workflow API

Import these objects from `siai`.

## `ImprovementResult`

```python
ImprovementResult(
    best: Candidate,
    score: float,
    evaluations: int,
    trials: int,
    journal: str,
)
```

The selected candidate, its mean search score, attempted candidate/seed
evaluations, evaluated configurations including the baseline, and journal path.

## `ImprovementLoop`

```python
ImprovementLoop(
    trainer,
    proposer,
    card: ModelCard,
    *,
    evaluation_budget: int = 9,
    seeds: tuple[int, ...] = (0, 1, 2),
    min_gain: float = 0.0,
    run_dir: str | Path = "siai-run",
)
```

### Trainer contract

```python
trainer.train(config: dict, *, seed: int, output_dir: Path) -> TrainingResult
```

Every call must create a fresh training run. The result supplies a finite reward,
observations, checkpoint path, and any named metrics required by model-card
guardrails.

### Proposer contract

```python
proposer.propose(request: ProposalRequest) -> Proposal
```

The proposal changes one top-level configuration lever. `IntelligentProposer`
implements this contract for any `generate(prompt) -> JSON text` provider adapter.

### `run`

```python
run(baseline: Candidate) -> ImprovementResult
```

Creates `run_dir` exclusively and evaluates the baseline. A baseline failure or
proposer error raises `RuntimeError` after recording evidence. Failed candidates
are rejected. A repeated candidate records a duplicate event and stops the run.
The next seed group is never started unless the remaining budget covers it.
Named metrics are averaged across the completed paired seed group. A candidate
must improve the primary reward and pass every guardrail to replace the incumbent.
Missing required metrics make the evaluation fail; the error remains in the
journal.
