# First experiment

This tutorial builds one improvement loop without a GPU or model-provider key.
It shows the same contracts used by a real training integration.

## 1. Define the model card

The model card states the goal, the success metric, the direction of improvement,
and the levers the proposer may change.

```python
from siai import ModelCard

card = ModelCard(
    objective="Improve validation accuracy within a fixed training budget",
    metric="accuracy",
    direction="maximize",
    levers={
        "learning_rate": "float greater than 0 and at most 0.01",
    },
    constraints=("Do not increase the number of training steps",),
)
```

`levers` is an allowlist. Its descriptions guide an intelligent proposer, while
the trainer adapter must enforce the real value bounds before expensive work.

## 2. Capture feedback

Observations are compact, identified pieces of search-time evidence.

```python
from siai import Candidate, Observation, ProposalRequest

incumbent = Candidate({"learning_rate": 0.001})

request = ProposalRequest(
    card=card,
    incumbent=incumbent,
    observations=(
        Observation(
            id="validation-accuracy",
            kind="task",
            summary="Accuracy after the baseline training run",
            value=0.71,
        ),
        Observation(
            id="loss-oscillation",
            kind="training",
            summary="Training loss oscillated during the final 20% of steps",
        ),
    ),
    remaining_evaluations=6,
)
```

Use validation or search data here. Never give the proposer feedback from the
final test set.

## 3. Connect a model provider

`IntelligentProposer` needs one function: prompt text in, JSON text out.

```python
from siai import IntelligentProposer


def generate(prompt: str) -> str:
    # Call your provider here and return only the response text.
    return """{
      "lever": "learning_rate",
      "value": 0.0005,
      "hypothesis": "A lower rate may reduce the observed loss oscillation",
      "evidence_ids": ["loss-oscillation", "validation-accuracy"],
      "expected_effect": "Potential validation accuracy improvement",
      "verification": "Compare paired seeds and check accuracy regressions"
    }"""


proposal = IntelligentProposer(generate).propose(request)
candidate = proposal.candidate(incumbent)
print(candidate.config)
```

SIAI rejects malformed output, unknown levers, invented evidence identifiers,
and configurations already present in the request history.

## 4. Train and decide

Train the candidate from the same initial checkpoint and compare it on the same
search split and paired seeds. Promote it only when the model-card reward improves.
Run the final test evaluation once, after the complete search has selected its
winner.

For automated selection and journaling, pass a candidate-producing callback to
[`Experiment`](reference/core.md#experiment). For language-model SFT,
continue with [Connect a trainer](guides/training.md).

## What you have built

You now have a stable boundary between:

1. task-specific training and measurement;
2. normalized evidence supplied to the proposer;
3. one validated, testable configuration change; and
4. budgeted selection based on a fixed reward.
