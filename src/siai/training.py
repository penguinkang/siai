"""Optional training integrations. Importing SIAI never imports GPU libraries."""
from dataclasses import dataclass
import math
from pathlib import Path
import time
from .proposer import Observation


@dataclass(frozen=True)
class TrainingResult:
    reward: float
    observations: tuple[Observation, ...]
    checkpoint: str


class SFTAdapter:
    """Shared TRL/Unsloth adapter. Run different backends in separate processes.

    factory(config, seed, output_dir) creates a FRESH trainer for every trial.
    scorer(trainer) returns the model-card reward on search data only.
    """
    def __init__(self, factory, scorer):
        self.factory, self.scorer = factory, scorer

    def train(self, config, *, seed, output_dir):
        unknown = set(config) - {'learning_rate', 'warmup_ratio'}
        if unknown:
            raise ValueError(f'unsupported training levers: {sorted(unknown)}')
        for key, value in config.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'invalid {key}')
            if key == 'learning_rate' and not 0 < value <= .01:
                raise ValueError('learning_rate must be in (0, .01]')
            if key == 'warmup_ratio' and not 0 <= value <= 1:
                raise ValueError('warmup_ratio must be in [0, 1]')
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=False)
        trainer = self.factory(dict(config), seed, str(path))
        start = time.monotonic()
        trainer.train()
        reward = float(self.scorer(trainer))
        if not math.isfinite(reward):
            raise ValueError('non-finite task reward')
        observations = [Observation('reward', 'task', 'Model-card search reward', reward),
                        Observation('runtime', 'cost', 'Training and scoring seconds', time.monotonic()-start)]
        for index, row in enumerate(trainer.state.log_history):
            for key in ('loss', 'eval_loss', 'grad_norm', 'learning_rate'):
                value = row.get(key)
                if isinstance(value, (float, int)) and math.isfinite(value):
                    observations.append(Observation(f'log-{index}-{key}', 'training',
                                                    f'{key} at step {row.get("step", "unknown")}', float(value)))
        checkpoint = str(path / 'checkpoint')
        trainer.save_model(checkpoint)
        return TrainingResult(reward, tuple(observations), checkpoint)


def sft_factory(backend, model_name, train_dataset, *, max_steps=10, max_length=256):
    """Create an optional backend factory with a fixed training budget.

    Plain TRL uses full fine tuning; Unsloth uses LoRA. These exercise the same
    adapter contract but are NOT compute-equivalent optimization baselines.
    """
    if backend not in ('trl', 'unsloth'):
        raise ValueError('backend must be trl or unsloth')
    def create(config, seed, output_dir):
        # Unsloth must initialize its patches before importing TRL/Transformers.
        if backend == 'unsloth':
            from unsloth import FastLanguageModel
        from transformers import set_seed
        from trl import SFTConfig, SFTTrainer
        set_seed(seed)
        args = SFTConfig(output_dir=output_dir, max_steps=max_steps,
                         max_length=max_length, per_device_train_batch_size=1,
                         logging_steps=1, report_to='none', seed=seed,
                         save_strategy='no', **config)
        if backend == 'trl':
            return SFTTrainer(model=model_name, args=args, train_dataset=train_dataset)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name, max_seq_length=max_length, load_in_4bit=False)
        model = FastLanguageModel.get_peft_model(
            model, r=8, lora_alpha=16, random_state=seed,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            use_gradient_checkpointing='unsloth')
        return SFTTrainer(model=model, processing_class=tokenizer, args=args,
                          train_dataset=train_dataset)
    return create
