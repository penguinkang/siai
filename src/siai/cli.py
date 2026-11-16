import argparse
import tempfile
from pathlib import Path
from .core import Candidate, Experiment


def main():
    parser = argparse.ArgumentParser(description="SIAI experiment kernel")
    parser.add_argument("command", choices=["demo"])
    parser.add_argument("--journal", default="demo.jsonl")
    args = parser.parse_args()
    # A tiny fitted model: optimize a scalar prediction on search observations.
    def evaluate(config, seed, split):
        targets = [1.8, 2.0, 2.2] if split == "search" else [1.9, 2.1]
        return -sum((config["weight"] - y) ** 2 for y in targets) / len(targets)
    def propose(best, history):
        return Candidate({"weight": len(history) * 0.5}, "increase scalar model weight")
    result = Experiment(evaluate, propose, budget=18, journal=args.journal).run(Candidate({"weight": 0}))
    print(f"Best model: {result.best.config}; search score: {result.score:.4f}")
    print(f"Final test score: {evaluate(result.best.config, 0, 'test'):.4f}")
