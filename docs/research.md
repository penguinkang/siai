# Research and positioning

Research cutoff: November 16, 2026. This is a targeted primary-source review,
not an exhaustive systematic review. Preprint findings are provisional; reported
results below are author claims, not independent replications.

## Landscape

| System / source | Improvement target and mechanism | Implication for SIAI |
| --- | --- | --- |
| [Autoresearch](https://github.com/karpathy/autoresearch) | Compact agent-driven single-GPU training research loop | Preserve the simple propose/evaluate/select interface; generalize task integration |
| [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) (2025) | LLM-generated programs selected by automated evaluators | Executable verification is central; improving training infrastructure differs from updating model weights |
| [AlphaEvolve update](https://deepmind.google/blog/alphaevolve-impact/) (May 2026) | Applications across scientific and computing domains | Domain-specific evaluators matter more than a universal reward |
| [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) (2025; revised March 2026) | Archive-based evolution of agent implementation | Search over an agent scaffold is distinct from parameter learning |
| [GEPA](https://arxiv.org/abs/2507.19457) (2025; revised February 2026) | Reflective prompt evolution using trajectories and a Pareto frontier | Prompts and feedback can be optimization artifacts; do not recreate GEPA as a novelty claim |
| [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) | Open-ended, sample-efficient program evolution | General code evolution is already an occupied open-source category |
| [Self-Improvements survey](https://arxiv.org/abs/2607.13104) (July 2026) | Organizes updates to parameters and scaffold components | Explicitly declare what changes and what stays fixed |
| [Red Queen Gödel Machine](https://arxiv.org/abs/2606.26294) (June; revised October 2026) | Co-evolves agents and evaluators under changing utilities | Changing evaluators require versioned evaluation epochs; scores across epochs are not directly comparable |
| [Does Failure Feedback Direct Agent Self-Modification?](https://arxiv.org/abs/2610.14662) (October 2026) | Analysis finds no detectable target-specific advantage across 145 eligible DGM modifications | Selection gains alone do not establish causal repair or transfer |
| [Topic-disjoint reflective optimization](https://arxiv.org/abs/2610.12601) (October 2026) | Evaluates deep-research prompt optimization with topic-disjoint splits | Separate adaptation, selection, and final reporting data |

## Synthesis

The field spans at least three different claims: adapting an artifact (prompt,
code, data recipe), training weights, and improving the improvement mechanism.
They should not share an undifferentiated “recursive self-improvement” label.
The visible trajectory is toward richer feedback, population search, reusable
skills, and changing evaluators. A parallel methodological concern is whether
reported gains survive unseen tasks, seeds, and equal-compute comparisons.

Our inference: another generic evolutionary agent runner has weak differentiation.
A small interoperability and evidence layer for model-training experiments is a
more defensible engineering contribution. Its research novelty remains unproven.

## Proposed contribution

SIAI: budgeted, auditable improvement experiments for models and their training
recipes. The initial user is an ML researcher with an existing training/evaluation
function who wants to replace manual trial selection with an inspectable loop.
The unit of improvement is a JSON configuration pointing to a model/training
recipe; a callback may train weights and return a score. SIAI does not itself
implement a trainer or claim autonomous recursive improvement.

The differentiator to validate is portable evidence: fixed evaluation contracts,
paired seeds, explicit budgets, candidate fingerprints, failure records and
lineage. These ingredients are individually established practices. Combining
them is an engineering hypothesis, not a new optimization algorithm.

## Required research validation before making performance claims

Compare incumbent-preserving random search, a fixed scripted proposer, an LLM
proposer, and a GEPA or ShinkaEvolve integration under equal training compute and
proposal cost. Use at least three task families (small-model training, adaptation,
and inference configuration), multiple independent search runs, and disjoint
final evaluation data. Report all failures, wall time, token cost, compute,
paired per-seed differences and confidence intervals. Keep the final test set
outside the proposer context and evaluate the selected artifact once per run.
Ablate reflection, archive selection and seed pairing separately. Publish null
results. Test evaluator tampering and reward exploitation when executing code.

## Scope and next releases

0.1 implements the minimal local kernel, not the full research agenda.
Next: persisted checkpoint manifests and evaluator/data version digests;
process-isolated runners with resource limits; an autoresearch adapter;
LLM proposer adapters; immutable evaluator services; final-report protocol;
paired uncertainty estimates and population search. Weight training, remote
execution and statistically justified promotion need separate implementation
and validation. A callback split label is not a security boundary.
