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
uses SFTTrainer directly; Unsloth remains covered only by mocked contract tests.
No claim of cross-backend training equivalence or GPU compatibility is made.
