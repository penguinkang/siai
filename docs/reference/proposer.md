# Proposer API

Import these objects from `siai`.

## `ModelCard`

```python
ModelCard(
    objective: str,
    metric: str,
    direction: str,
    levers: dict[str, str],
    constraints: tuple[str, ...] = (),
)
```

`direction` is `"maximize"` or `"minimize"`. At least one lever is required.
Lever descriptions and constraints guide the proposer; the trainer must enforce
them.

## `Observation`

```python
Observation(
    id: str,
    kind: str,
    summary: str,
    value: float | None = None,
    artifact: str | None = None,
)
```

Identified search-time evidence. Numeric values must be finite. An artifact is an
opaque reference and is not fetched by SIAI.

## `ProposalRequest`

```python
ProposalRequest(
    card: ModelCard,
    incumbent: Candidate,
    observations: tuple[Observation, ...],
    history: tuple[dict, ...] = (),
    remaining_evaluations: int = 1,
)
```

The complete input contract for an intelligent proposer.

## `Proposal`

```python
Proposal(
    lever: str,
    value: object,
    hypothesis: str,
    evidence_ids: tuple[str, ...],
    expected_effect: str,
    verification: str,
)
```

`candidate(incumbent)` returns a new `Candidate` by replacing the selected
top-level lever while preserving all other configuration values.

## `IntelligentProposer`

```python
IntelligentProposer(generate)
```

`generate(prompt: str) -> str` supplies a provider-specific model call. Calling
`propose(request)` makes exactly one call and returns a validated `Proposal`.
It raises `ValueError` for exhausted budget, duplicate observation IDs, malformed
JSON contracts, unknown levers, invented evidence IDs, and repeated candidates.
