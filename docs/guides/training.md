# Connect a trainer

`SFTAdapter` provides one result contract for TRL and Unsloth-style SFT runs.
Heavy training dependencies remain optional and are imported only when a backend
factory is used.

## Adapter contract

```python
from siai.training import SFTAdapter, sft_factory

factory = sft_factory(
    backend="trl",
    model_name="your-model",
    train_dataset=train_dataset,
    max_steps=100,
    max_length=512,
)


def score(trainer):
    metrics = evaluate_on_search_split(trainer.model)
    return metrics["exact_match"]


def guardrail_metrics(trainer):
    metrics = evaluate_on_search_split(trainer.model)
    return {
        "latency_ms": metrics["latency_ms"],
        "valid_output_rate": metrics["valid_output_rate"],
    }


adapter = SFTAdapter(factory, score, metric_scorer=guardrail_metrics)
result = adapter.train(
    {"learning_rate": 2e-4, "warmup_ratio": 0.1},
    seed=0,
    output_dir="runs/trial-001",
)
```

`TrainingResult` contains the finite task reward, named search metrics,
normalized observations from trainer logs, and the saved checkpoint path.

## Backend selection

Use `backend="trl"` for the plain TRL path and `backend="unsloth"` for the
Unsloth model-loading and LoRA path. Run the backends in separate processes:
Unsloth patches shared training libraries during import.

!!! warning "Not compute-equivalent"
    The current plain TRL factory uses full fine-tuning while the Unsloth factory
    uses LoRA. They demonstrate a shared interface, not a fair performance
    comparison.

## Adapter responsibilities

Your production adapter should also pin the model, tokenizer, dataset, evaluator,
environment, and initial checkpoint. It should validate every lever before GPU
work and keep final-test credentials outside the search process.

The current built-in adapter allows only `learning_rate` and `warmup_ratio`.
## Validated compatibility

| Path | Environment | Validation |
| --- | --- | --- |
| TRL `SFTTrainer` | macOS, PyTorch/MPS | Full workflow with a local tiny model |
| Unsloth `UnslothTrainer` | Apple Silicon, MLX | Full LoRA workflow with SmolLM2-135M |

Exact versions and measured outputs live on `dev/micro-projects`. These runs
validate the shared contract; they are not performance comparisons. CUDA and
multi-GPU environments remain unvalidated.
