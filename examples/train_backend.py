"""Run separately per backend: python examples/train_backend.py trl --output trial-trl"""
import argparse
from siai.training import SFTAdapter, sft_factory


def main():
    p = argparse.ArgumentParser()
    p.add_argument('backend', choices=['trl', 'unsloth'])
    p.add_argument('--model', default='Qwen/Qwen3-0.6B')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    # Initialize Unsloth before any imports that load Transformers.
    if args.backend == 'unsloth':
        import unsloth
    from datasets import Dataset
    train = Dataset.from_dict({'text': ['The capital of France is Paris.'] * 8})
    validation = Dataset.from_dict({'text': ['The capital of Japan is Tokyo.']})
    def score(trainer):
        # Smoke-test reward only; replace with model-card task scoring in research.
        return -trainer.evaluate(eval_dataset=validation)['eval_loss']
    result = SFTAdapter(sft_factory(args.backend, args.model, train, max_steps=2), score).train(
        {'learning_rate': .0001, 'warmup_ratio': 0.1}, seed=0, output_dir=args.output)
    print(result)

if __name__ == '__main__': main()
