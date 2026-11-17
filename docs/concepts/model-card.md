# Design a model card

The model card is the optimization boundary. A precise card makes proposals more
useful and prevents the improvement loop from silently changing its own goal.

```python
from siai import ModelCard

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
)
```

## Objective

Describe the behavior to improve and its operating boundary. Include material
cost, latency, or quality limits.

## Metric and direction

Name one primary selection metric. SIAI's core experiment runner maximizes scores;
normalize minimized metrics, such as loss, by negating them at the integration
boundary.

## Levers

Expose only values the training adapter can validate and apply. One proposal
changes one top-level lever. If a change must be atomic across several settings,
represent it as a named recipe lever.

## Constraints

Constraints in `ModelCard` are guidance to the proposer. They are not executable
validation. Enforce numeric ranges, resource limits, and forbidden changes in the
trainer adapter before training starts.

