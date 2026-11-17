# The improvement loop

SIAI separates model improvement into four contracts.

## Measure

A trainer starts from a controlled initial state, trains one configuration, and
returns a model-card reward plus diagnostic observations. Reward decides whether
the candidate improved. Diagnostics help explain what to try next.

## Propose

The proposer receives the incumbent, model card, observations, history, and
remaining budget. It returns one lever change with a falsifiable hypothesis and
citations to the evidence it used.

## Select

Candidates are compared under paired seeds. A strict mean-score improvement over
the incumbent is required. Failed trials are recorded and rejected.

## Record

Every evaluated configuration has a content fingerprint and parent identifier.
The JSONL journal records scores, errors, decisions, and cumulative evaluation
count.

!!! warning "Selection is not generalization"
    A better validation score does not establish causal repair or performance on
    unseen tasks. Use disjoint final-test data, multiple independent searches,
    equal compute budgets, and uncertainty estimates before making effectiveness
    claims.

## Budget semantics

An evaluation is one candidate/seed pair. The baseline consumes budget. A seed
group starts only when enough budget remains to run the whole group. If a trial
fails partway through, attempted evaluations still count.

