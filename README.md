# SIAI

A small open-source kernel for budgeted model-improvement experiments.
Python 3.11+, no runtime dependencies. Prototype version 0.1.

```sh
python -m pip install -e .
siai demo --journal demo.jsonl
python -m unittest discover -s tests -v
```

```python
from siai import Candidate, Experiment

# Replace with a training/evaluation function; maximize the returned score.
def evaluate(config, seed, split):
    return -(config['weight'] - 2) ** 2

def propose(best, history):
    return Candidate({'weight': len(history) * 0.5}, 'Increase model weight')

result = Experiment(evaluate, propose, budget=18, journal='run.jsonl').run(
    Candidate({'weight': 0}))
print(result.best.config)
```

The included demo fits a scalar prediction model, requires no GPU or API key,
and reports a separate final score. It is a functional example, not evidence of
LLM-driven discovery. Training real models and generating proposals are callback
integrations supplied by the application.

Features: paired seeds, bounded evaluation calls, strict mean-score promotion,
configuration fingerprints, parent lineage and JSONL failure records.
Callbacks are trusted local Python. This release provides neither code isolation
nor hard wall-time/compute limits. Final holdouts must be managed independently.

Read the [research review](docs/research.md) for the November 16, 2026 landscape,
positioning and proposed validation study, and the [design](docs/design.md) for
contracts and limitations. No benchmark superiority or recursive capability is
claimed. Contributions should include a reproducible example and relevant tests.

## Intelligent proposer

The proposer is a standalone, trainer-independent component: one structured
request, one model call, one proposed lever change. Supply any model provider
through a `generate(prompt) -> JSON text` callable.

```python
from siai import ModelCard, Observation, ProposalRequest, IntelligentProposer

request = ProposalRequest(
    card=ModelCard(
        objective='Improve held-out QA accuracy within the training budget',
        metric='accuracy', direction='maximize',
        levers={'learning_rate': 'float between 0.00001 and 0.01'},
        constraints=('Do not increase training steps',)),
    incumbent=Candidate({'learning_rate': 0.001}),
    observations=(Observation('train-loss', 'training', 'Loss oscillates'),),
    remaining_evaluations=6,
)
# generate is your provider adapter, returning the documented JSON fields.
# proposal = IntelligentProposer(generate).propose(request)
# next_config = proposal.candidate(request.incumbent).config
```

Input carries the model-card objective, direction, editable levers, constraints,
incumbent, search observations, history and remaining budget. Observations can
represent reward slices, loss curves summarized by the trainer, gradient issues,
error examples, resource measurements or prior failures. Attach opaque artifact
references for traceability; the proposer does not load them automatically.

Output specifies one lever/value, a hypothesis, cited observation IDs, an expected
effect and a verification plan. Unknown levers, invented evidence IDs, malformed
responses and repeated configurations are rejected. Evidence citations establish
traceability, not correctness or causality. Numeric/domain constraints remain the
trainer adapter's responsibility; constraints expressed in prose are not enforced.

The intelligent proposer is not automatically wired into `Experiment`: an adapter
must collect observations and construct requests, persist the complete proposal,
and return its candidate. The current scalar kernel maximizes reward; negate a
loss when integrating a minimizing model-card metric. Never include final-test
feedback in requests. No provider-specific dependency or API key is required by
the library itself.

The selected v0.1.0 training integration target is **TRL SFTTrainer**, with
Unsloth considered later as an acceleration option. See the
[PoC specification](docs/poc.md) for the task, reward, feedback signals and
acceptance criteria. The TRL adapter is not implemented yet.

## Plug-and-train adapters

` siai.training.SFTAdapter ` now supports a shared training/result contract with
` sft_factory('trl', ...) ` and ` sft_factory('unsloth', ...) `. Both return task
reward, training observations and a saved checkpoint. Supply your own scorer to
implement the model-card metric. Each call constructs a fresh trainer; trial
output directories must be new. Only learning_rate and warmup_ratio are editable.

Install each backend using its official environment instructions, then install
SIAI into that environment. Run backends in separate processes because Unsloth
patches shared training libraries. Dependencies are optional and not pinned yet:
there is no verified dependency compatibility matrix for this prototype.

The development branch `dev/micro-projects` contains sample training projects,
backend smoke scripts, measured results and environment locks. These assets are
excluded from `master` and release artifacts. The core adapter has contract-test
coverage; backend training validation is maintained on that development branch.
