# Build a proposer

The proposer is deliberately provider-neutral. Supply a callable with this shape:

```python
generate(prompt: str) -> str
```

The returned string must be a JSON object with exactly these fields:

| Field | Meaning |
| --- | --- |
| `lever` | One allowlisted top-level configuration key |
| `value` | Replacement value for that lever |
| `hypothesis` | Falsifiable reason the change may improve reward |
| `evidence_ids` | Non-empty list of known observation IDs |
| `expected_effect` | Qualitative prediction, without guaranteed claims |
| `verification` | Test plan including relevant regression checks |

## Provider adapter

```python
import json
from siai import IntelligentProposer


def generate(prompt):
    response = provider.generate(
        prompt=prompt,
        response_format="json",
        temperature=0.2,
    )
    return response.text


proposer = IntelligentProposer(generate)
```

Keep provider credentials and retry policy in the application. Persist the raw
request, raw response, provider model identifier, and proposal before training.

## Feedback quality

Useful observations include:

- primary reward and important slices;
- loss trajectory and gradient diagnostics;
- malformed-output and task-error categories;
- runtime, memory, tokens, and monetary cost; and
- failures from previous configurations.

Summaries should state what was measured. Do not turn speculation into a measured
observation. Artifact references are opaque identifiers; SIAI does not fetch them.

## Security boundary

Treat feedback and artifact text as untrusted data. Prompt instructions reduce
accidental misuse but do not provide isolation. Do not give the proposer final-test
results, evaluator credentials, or permission to modify the evaluation service.

