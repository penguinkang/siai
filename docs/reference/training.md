# Training API

Import these objects from `siai.training`. Trainer dependencies are optional and
are loaded only when a backend factory creates a trainer.

## `TrainingResult`

```python
TrainingResult(
    reward: float,
    observations: tuple[Observation, ...],
    checkpoint: str,
    metrics: dict[str, float] = {},
)
```

The finite model-card reward, extracted trainer observations, and saved checkpoint
path. `metrics` contains finite named search metrics used by model-card
guardrails.

## `SFTAdapter`

```python
SFTAdapter(factory, scorer, metric_scorer=None)
```

`factory(config, seed, output_dir)` must create a fresh trainer. `scorer(trainer)`
must return the finite search reward.
When supplied, `metric_scorer(trainer)` returns a mapping of finite metric names
to values. It runs after training alongside the primary scorer.

```python
adapter.train(config, *, seed: int, output_dir: str | Path) -> TrainingResult
```

The output directory must not exist. The built-in adapter accepts only
`learning_rate` in `(0, 0.01]` and `warmup_ratio` in `[0, 1]`. It calls
`trainer.train()`, extracts supported finite log values, and saves the model to
an output `checkpoint` directory.

## `sft_factory`

```python
sft_factory(
    backend: str,
    model_name: str,
    train_dataset,
    *,
    max_steps: int = 10,
    max_length: int = 256,
)
```

Returns a trainer factory. `backend` is `"trl"` or `"unsloth"`. The TRL path
uses full fine-tuning. The Unsloth path loads a fast language model and attaches
LoRA adapters. Run different backends in separate processes.
