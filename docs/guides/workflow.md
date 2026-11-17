# Run the full loop

`ImprovementLoop` connects a trainer, proposer, and model card. It trains the
baseline first, proposes one change at a time, evaluates every candidate with
paired seeds, and keeps only improvements.

```python
from siai import Candidate, ImprovementLoop, IntelligentProposer, ModelCard
from siai.training import SFTAdapter


card = ModelCard(
    objective="Improve held-out extraction accuracy",
    metric="exact_match",
    direction="maximize",
    levers={
        "learning_rate": "float in (0, 0.01]",
        "warmup_ratio": "float in [0, 1]",
    },
    constraints=("Keep optimizer steps and training data fixed",),
)

trainer = SFTAdapter(factory, score_search_split)
proposer = IntelligentProposer(generate)

result = ImprovementLoop(
    trainer,
    proposer,
    card,
    evaluation_budget=9,
    seeds=(0, 1, 2),
    min_gain=0.0,
    run_dir="runs/extraction-001",
).run(
    Candidate({"learning_rate": 2e-4, "warmup_ratio": 0.1})
)

print(result.best.config)
print(result.score)
print(result.journal)
```

## What happens during a run

1. The baseline is trained once per seed.
2. Trainer observations are given stable trial-and-seed-prefixed IDs.
3. The proposer receives the incumbent evidence, complete trial history, and
   remaining evaluation budget.
4. The proposed configuration is trained from fresh state with the same seeds.
5. Mean model-card reward determines promotion or rejection.
6. The complete proposal, scores, observations, checkpoints, errors, and decision
   are appended to `journal.jsonl`.

When a rejected candidate fails, its failure observation is supplied alongside
the incumbent evidence for the next proposal. A failed baseline aborts because
there is no valid incumbent. In both cases, the evidence remains in the journal.

## Metric direction

Unlike the low-level `Experiment`, `ImprovementLoop` honors the model card's
`direction`. It promotes higher scores for `maximize` and lower scores for
`minimize`. `min_gain` is an absolute threshold in the metric's own units.

## Evaluation boundaries

The workflow operates on search or validation reward only. Evaluate
`result.best` on the isolated final-test split after the run. The final test must
not appear in observations, proposal history, or trainer scoring during search.

## Run directories

The run directory must not already exist. Each trial gets a directory named
`trial-NNN/seed-S`. This prevents accidental overwrites and makes every checkpoint
traceable to a candidate and seed. Resume is not supported in v0.1.0.

