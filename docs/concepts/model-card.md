# Design a model card

The model card is the optimization boundary. A precise card makes proposals more
useful and prevents the improvement loop from silently changing its own goal.

```python
from siai import MetricGuardrail, ModelCard

card = ModelCard(
    objective="Improve exact-match extraction quality within 30 GPU-minutes",
    metric="exact_match",
    direction="maximize",
    levers={
        "learning_rate": "float in (0, 0.01]",
        "warmup_ratio": "float in [0, 1]",
    },
    constraints=(
        "Keep training examples and optimizer steps fixed",
        "Do not modify the evaluator or decoding policy",
    ),
    guardrails=(
        MetricGuardrail("latency_ms", "at_most", 100),
        MetricGuardrail("valid_output_rate", "at_least", 0.99),
    ),
)
```

## Objective

Describe the behavior to improve and its operating boundary. Include material
cost, latency, or quality limits.

## Metric and direction

Name one primary selection metric and choose `maximize` or `minimize`.
`ImprovementLoop` applies that direction without requiring negated values.

## Guardrails

Guardrails are named search metrics that determine whether a candidate is
eligible for promotion. Each uses an inclusive `at_least` or `at_most`
threshold. Trainers must report every required metric for every seed. SIAI
averages each metric across the paired seed group and rejects an otherwise
better candidate when a threshold fails.

Use guardrails for measurable limits such as latency, invalid-output rate,
memory, fairness slices, or cost. Keep final-test metrics outside the search
process.

## Levers

Expose only values the training adapter can validate and apply. One proposal
changes one top-level lever. If a change must be atomic across several settings,
represent it as a named recipe lever.

## Constraints

Free-text constraints guide the proposer. Guardrails and trainer lever validation
are executable. Enforce hard resource limits and forbidden operations in the
trainer or process runner before training starts.
