# End-to-end micro projects

Run from the repository root (Python 3.12):

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r examples/micro/requirements.lock.txt
PYTHONPATH=src .venv/bin/python examples/micro/run.py --output examples/micro/results.json
```

All four projects run on CPU without downloaded datasets or model weights:

- sklearn decision tree on noisy two-moons classification; change tree depth.
- PyTorch MLP on synthetic binary classification; change learning rate.
- PyTorch transformer encoder on synthetic sequence classification; change learning rate.
- TRL SFTTrainer on a tiny GPT-2 architecture: two local pretraining steps followed
  by four fine-tuning steps from the same initial checkpoint; change learning rate.

Each project trains a baseline, records search reward as an observation, constructs
an IntelligentProposer request with a model card, validates a proposal, trains the
candidate, selects using search reward, and scores the selected model on a final
split. The proposal provider is a deterministic JSON fixture. No API credential
is required. This tests orchestration, not learned proposal intelligence.

The first three rewards are accuracy; the LM reward is negative cross-entropy.
LM search/test sequences are distinct but intentionally trivial. The local model
is not representative of a useful pretrained language model. Poor baseline
settings and one selected intervention make these suitable smoke tests, not
comparative evidence of optimization performance. One fixed seed is used.

results.json contains measured outputs; requirements.lock.txt records the local
resolved package versions. Tests exercise actual sklearn, PyTorch and TRL, but
do not yet route every project through a common trainer adapter. The LM project
uses SFTTrainer directly. The separate full-workflow examples exercise the
shared adapter with both TRL and Unsloth. No claim of cross-backend training
equivalence or GPU compatibility is made.

## Full TRL workflow

`trl_workflow.py` exercises the current `master` contracts with actual CPU
training: `SFTAdapter` → `ImprovementLoop` → `IntelligentProposer` →
`OpenAIResponsesGenerator` → candidate training → selection → isolated final
evaluation.

```bash
PYTHONPATH=src .venv/bin/python examples/micro/trl_workflow.py
```

The base model and tokenizer are created locally and every candidate starts from
the same checkpoint. The provider transport is an injected deterministic
Responses API fixture because this project must run without credentials. It
validates the strict structured-output request and metadata path, but does not
validate a live OpenAI request or proposer intelligence. Measured output is in
`trl-workflow-result.json`; checkpoints and the full JSONL journal remain in a
temporary directory and are not committed.

## Full Unsloth workflow on Apple Silicon

`unsloth_workflow.py` exercises the same package contracts with Unsloth's MLX
trainer and a downloaded 135M parameter SmolLM2 checkpoint. Install current
Unsloth in a separate Python 3.13 environment, then run:

```bash
PYTHONPATH=src /path/to/unsloth-venv/bin/python examples/micro/unsloth_workflow.py
```

The current MLX trainer trains, logs, and saves through the shared adapter, but
does not expose TRL's public `evaluate()` method. The injected scorer therefore
computes held-out cross-entropy directly from `trainer.model` and
`trainer.tokenizer`. Measured output and exact versions are recorded in
`unsloth-workflow-result.json`.
