# Design

Data flow: incumbent configuration → proposer → candidate → fixed seeded
evaluator → finite scores → mean-score promotion → JSONL evidence.

The standard-library-only kernel accepts two Python callbacks. Model providers,
training libraries and search strategies remain outside the core. Configurations
must be JSON serializable and are copied across callback boundaries. SHA-256
fingerprints identify configuration content; they do not identify external
weights, datasets or evaluator code. Every candidate has a hypothesis and parent.
The low-level `Experiment` maximizes scores; `ImprovementLoop` applies the model
card's maximize or minimize direction. A strict minimum gain threshold handles
ties but provides no statistical guarantee. Named guardrail metrics can prevent
promotion, but their paired means do not establish statistical significance.

The budget counts attempted candidate/seed evaluations, including baseline and
failures. Incomplete seed groups are not launched. Duplicate configurations stop
the run. Baseline failures abort; later failures are recorded and rejected.
Journals use exclusive creation to prevent overwrites. There is no resume,
concurrent writer support, timeout or crash-atomic checkpointing in 0.1.

Trust model: callbacks execute with the host process's permissions. Do not run
untrusted generated code through this runner. Search history contains only search
results; final evaluation is the application's responsibility. Strong holdout
isolation requires a separate service or process and restricted credentials.

A future artifact manifest should bind configuration, weights, code, evaluator,
data split and environment versions. Until then, reproducibility depends on the
application pinning these external inputs. An append-only journal is useful
evidence, but is not tamper-proof provenance.

## Proposer-first direction

The central abstraction is now `ProposalRequest -> Proposal`. The experiment
runner is one possible consumer. The request describes the objective rather than
assuming every trainer shares the same reward semantics. Observations preserve
IDs, kinds, summaries, optional values and artifact references; upstream adapters
are responsible for truthful aggregation, units and missing-data handling.
The response changes one top-level configuration lever to keep experiments easy
to interpret. Complex coupled interventions can later use a named recipe lever.

The initial strategy uses one model call with a bounded decision packet. Current
evidence is ranked deterministically, and journal history is reduced to tested
lever outcomes with direction-normalized gains. There is no learned lever ranking
or calibrated expected reward estimator yet.
Effectiveness must be measured against random and scripted proposal baselines;
format validation alone is insufficient. Future work should add persisted
artifact manifests, typed lever constraints, stopping/abstention and an
equal-cost proposer benchmark. Avoid adding multi-agent orchestration before it
demonstrates additional reward per unit cost. Feedback text is untrusted input;
prompt instructions alone do not provide security isolation.
