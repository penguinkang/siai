# Training API

Import these objects from `siai.training`. Trainer dependencies are optional and
are loaded only when a backend factory creates a trainer.

## `TrainingResult`

```python
TrainingResult(
    reward: float,
    observations: tuple[Observation, ...],
    checkpoint: str,
)
```

The finite model-card reward, extracted trainer observations, and saved checkpoint
path.

## `SFTAdapter`

```python
SFTAdapter(factory, scorer)
```

`factory(config, seed, output_dir)` must create a fresh trainer. `scorer(trainer)`
must return the finite search reward.

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
