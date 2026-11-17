# SIAI

Budgeted, auditable improvement loops for machine-learning models.

SIAI connects trainer feedback to an intelligent proposer, tests one controlled
configuration change, and keeps it only when the model-card reward improves. The
proposal contract stays stable across trainers, fine-tuning libraries, and
feedback systems.

> **Prototype status:** v0.1.0 is under active development. No benchmark
> superiority or recursive capability is claimed.

## Quickstart

```bash
python -m pip install -e .
siai demo --journal demo.jsonl
```

```python
from siai import Candidate, Experiment


def evaluate(config, seed, split):
    return -(config["weight"] - 2) ** 2


def propose(best, history):
    return Candidate(
        {"weight": best.config["weight"] + 0.5},
        "Move the weight toward the observed optimum",
    )


result = Experiment(
    evaluate,
    propose,
    budget=18,
    journal="run.jsonl",
).run(Candidate({"weight": 0.0}))

print(result.best.config)
```

## Documentation

- [First experiment](docs/getting-started.md)
- [Improvement-loop concepts](docs/concepts/improvement-loop.md)
- [Intelligent proposer guide](docs/guides/proposer.md)
- [Model-provider integration](docs/guides/providers.md)
- [TRL and Unsloth integration](docs/guides/training.md)
- [End-to-end improvement workflow](docs/guides/workflow.md)
- [API reference](docs/reference/core.md)
- [Research and positioning](docs/research.md)

Build the complete searchable documentation locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Development

```bash
python -m unittest discover -s tests -v
mkdocs build --strict
```

`master` contains the release library and documentation. The
`dev/micro-projects` branch contains sample training projects, measured results,
and larger training environments. Those assets are excluded from release
artifacts.

## Development training projects

This branch contains CPU-runnable integration projects and measured outputs that
are intentionally excluded from `master` and release artifacts.

```sh
python examples/train_backend.py trl --output trial-trl
python examples/train_backend.py unsloth --output trial-unsloth
```

The backend scripts exercise API interchangeability. Plain TRL performs full
fine-tuning and Unsloth uses LoRA, so their results are not compute-equivalent.

## Four actual end-to-end micro projects

See [examples/micro](examples/micro/README.md) for CPU-runnable sklearn, MLP,
transformer encoder and tiny-LM fine-tuning projects. Real runs and resolved
versions are recorded there. These supersede the earlier statement that no
training dependencies are installed: sklearn/PyTorch/TRL training now runs
locally. Actual Unsloth training remains unverified.

SIAI is licensed under the MIT License.
