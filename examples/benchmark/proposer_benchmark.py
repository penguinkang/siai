"""Equal-training-budget proposer benchmark with an optional live model.

The default offline model is a deterministic contract fixture. Its results test
the benchmark and context-ablation plumbing, not model intelligence.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import statistics
import tempfile

import numpy as np
import torch
from sklearn.datasets import make_classification, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from siai import (
    Candidate,
    ImprovementLoop,
    IntelligentProposer,
    ModelCard,
    Observation,
    OpenAIResponsesGenerator,
    Proposal,
)
from siai.core import canonical
from siai.training import TrainingResult


@dataclass
class Task:
    name: str
    family: str
    lever: str
    values: tuple
    baseline: object
    score: object
    executions: int = 0

    def evaluate(self, config, seed, split):
        self.executions += 1
        return float(self.score(config[self.lever], seed, split))


class CachedTask:
    def __init__(self, task):
        self.task = task
        self.cache = {}

    def evaluate(self, config, seed, split):
        key = (canonical(config), seed, split)
        if key not in self.cache:
            self.cache[key] = self.task.evaluate(config, seed, split)
        return self.cache[key]

    def __getattr__(self, name):
        return getattr(self.task, name)


def tree_task():
    x, y = make_moons(n_samples=360, noise=0.2, random_state=11)
    train_x, remainder_x, train_y, remainder_y = train_test_split(
        x, y, test_size=0.44, random_state=12, stratify=y
    )
    search_x, final_x, search_y, final_y = train_test_split(
        remainder_x, remainder_y, test_size=0.5, random_state=13, stratify=remainder_y
    )

    def score(depth, seed, split):
        model = DecisionTreeClassifier(
            max_depth=depth, splitter="random", random_state=seed
        ).fit(train_x, train_y)
        data_x, data_y = (search_x, search_y) if split == "search" else (final_x, final_y)
        return model.score(data_x, data_y)

    return Task("decision_tree", "parametric statistical ML", "depth", (1, 2, 3, 4, 6, 8), 1, score)


def mlp_task():
    generator = torch.Generator().manual_seed(21)
    x = torch.randn(240, 12, generator=generator)
    weights = torch.linspace(-1, 1, 12)
    y = ((x @ weights + 0.25 * torch.randn(240, generator=generator)) > 0).long()

    def score(learning_rate, seed, split):
        torch.manual_seed(seed)
        model = torch.nn.Sequential(
            torch.nn.Linear(12, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for _ in range(30):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x[:140]), y[:140])
            loss.backward()
            optimizer.step()
        start, end = (140, 190) if split == "search" else (190, 240)
        with torch.no_grad():
            return (model(x[start:end]).argmax(-1) == y[start:end]).float().mean().item()

    values = (0.00001, 0.0001, 0.001, 0.005, 0.01, 0.02)
    return Task("torch_mlp", "deep learning", "learning_rate", values, values[0], score)


def threshold_task():
    x, y = make_classification(
        n_samples=480,
        n_features=10,
        n_informative=6,
        weights=[0.68, 0.32],
        flip_y=0.08,
        random_state=31,
    )
    train_x, remainder_x, train_y, remainder_y = train_test_split(
        x, y, test_size=0.42, random_state=32, stratify=y
    )
    search_x, final_x, search_y, final_y = train_test_split(
        remainder_x, remainder_y, test_size=0.5, random_state=33, stratify=remainder_y
    )
    model = LogisticRegression(max_iter=500, random_state=0).fit(train_x, train_y)

    def score(threshold, seed, split):
        del seed
        data_x, data_y = (search_x, search_y) if split == "search" else (final_x, final_y)
        predictions = (model.predict_proba(data_x)[:, 1] >= threshold).astype(int)
        true_positive = ((predictions == 1) & (data_y == 1)).sum()
        false_positive = ((predictions == 1) & (data_y == 0)).sum()
        false_negative = ((predictions == 0) & (data_y == 1)).sum()
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        return 2 * precision * recall / max(precision + recall, 1e-12)

    values = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    return Task("classification_threshold", "inference configuration", "threshold", values, 0.5, score)


class TaskTrainer:
    def __init__(self, task):
        self.task = task

    def train(self, config, *, seed, output_dir):
        if config.get(self.task.lever) not in self.task.values:
            raise ValueError(f"{self.task.lever} must be one of {list(self.task.values)}")
        path = Path(output_dir)
        path.mkdir(parents=True)
        checkpoint = path / "checkpoint"
        checkpoint.mkdir()
        reward = self.task.evaluate(config, seed, "search")
        return TrainingResult(
            reward,
            (Observation("reward", "metric", "Measured search reward", reward),),
            str(checkpoint),
        )


class DirectProposer:
    def proposal(self, request, value, hypothesis):
        return Proposal(
            lever=next(iter(request.card.levers)),
            value=value,
            hypothesis=hypothesis,
            evidence_ids=(request.observations[0].id,),
            expected_effect="Potential search-reward improvement",
            verification="Compare paired seeds and preserve final-test isolation",
        )

    @staticmethod
    def tried_values(request, lever):
        return {
            row.get("config", {}).get(lever)
            for row in request.history
            if row.get("type") == "trial"
        }


class RandomProposer(DirectProposer):
    def __init__(self, values, seed):
        self.values = values
        self.random = random.Random(seed)

    def propose(self, request):
        lever = next(iter(request.card.levers))
        available = [value for value in self.values if value not in self.tried_values(request, lever)]
        return self.proposal(request, self.random.choice(available), "Random unexplored value")


class ScriptedProposer(DirectProposer):
    def __init__(self, values):
        self.values = values

    def propose(self, request):
        lever = next(iter(request.card.levers))
        tried = self.tried_values(request, lever)
        value = next(value for value in self.values if value not in tried)
        return self.proposal(request, value, "Next value in a fixed task-independent schedule")


class OfflineLeverModel:
    """Deterministic fixture shared by raw and compact context ablations."""

    def __init__(self, values):
        self.values = values

    def __call__(self, prompt):
        context = json.loads(prompt.split("REQUEST:\n", 1)[1])
        lever = next(iter(context["card"]["levers"]))
        rows = []
        if "experiments" in context:
            incumbent = context["incumbent"]["config"][lever]
            for row in context["experiments"]:
                value = row.get("config", {}).get(lever, row.get("value"))
                if value is not None and row.get("score") is not None:
                    rows.append((value, row["score"]))
        else:
            for row in context.get("history", []):
                if row.get("mean") is not None and lever in row.get("config", {}):
                    rows.append((row["config"][lever], row["mean"]))
            incumbent = context["incumbent"]["config"][lever]
        tested = {value for value, _ in rows} | {incumbent}
        if rows:
            best_value = max(rows, key=lambda item: item[1])[0]
            best_index = self.values.index(best_value)
            candidates = sorted(
                (value for value in self.values if value not in tested),
                key=lambda value: (abs(self.values.index(value) - best_index), self.values.index(value)),
            )
        else:
            candidates = [value for value in self.values if value not in tested]
            candidates.sort(key=lambda value: abs(self.values.index(value) - len(self.values) // 2))
        evidence_id = context["observations"][0]["id"]
        return json.dumps({
            "lever": lever,
            "value": candidates[0],
            "hypothesis": "Explore the nearest untested value around the strongest measured result",
            "evidence_ids": [evidence_id],
            "expected_effect": "Potential improvement over the incumbent search reward",
            "verification": "Compare paired seeds and check the isolated final split once",
        })


class CountingGenerator:
    def __init__(self, generate):
        self.generate = generate
        self.calls = 0
        self.prompt_chars = 0
        self.last_metadata = None

    def __call__(self, prompt):
        self.calls += 1
        self.prompt_chars += len(prompt)
        output = self.generate(prompt)
        self.last_metadata = getattr(self.generate, "last_metadata", None)
        return output


class RawHistoryProposer:
    def __init__(self, generate):
        self.generate = generate

    def propose(self, request):
        prompt = (
            "You propose one allowed lever change using only supplied evidence. "
            "Return the exact proposal JSON contract and cite an observation.\nREQUEST:\n"
            + canonical(asdict(request))
        )
        validator = IntelligentProposer(lambda _: self.generate(prompt))
        return validator.propose(request)


def model_generator(values, openai_model):
    if openai_model:
        return CountingGenerator(OpenAIResponsesGenerator(openai_model))
    return CountingGenerator(OfflineLeverModel(values))


def make_proposer(strategy, task, run_seed, openai_model):
    if strategy == "random":
        return RandomProposer(task.values, run_seed), None
    if strategy == "scripted":
        schedule = tuple(value for value in task.values if value != task.baseline)
        return ScriptedProposer(schedule), None
    generate = model_generator(task.values, openai_model)
    if strategy == "raw_model":
        return RawHistoryProposer(generate), generate
    return IntelligentProposer(generate), generate


def one_run(task, strategy, run_index, evaluations, seeds, openai_model):
    proposer, generator = make_proposer(strategy, task, 10_000 + run_index, openai_model)
    with tempfile.TemporaryDirectory() as temporary:
        result = ImprovementLoop(
            TaskTrainer(task),
            proposer,
            ModelCard(
                objective=f"Improve {task.name} search reward",
                metric="task_reward",
                direction="maximize",
                levers={task.lever: f"one of {list(task.values)}"},
                constraints=("Keep data, evaluator, and logical training budget fixed",),
            ),
            evaluation_budget=evaluations,
            seeds=seeds,
            run_dir=Path(temporary) / "run",
        ).run(Candidate({task.lever: task.baseline}))
        rows = [json.loads(line) for line in Path(result.journal).read_text().splitlines()]
    final_score = task.evaluate(result.best.config, 50_000 + run_index, "final")
    return {
        "run": run_index,
        "baseline_search_reward": rows[0]["mean"],
        "selected_search_reward": result.score,
        "search_gain": result.score - rows[0]["mean"],
        "final_test_reward": final_score,
        "selected_config": result.best.config,
        "logical_evaluations": result.evaluations,
        "trials": result.trials,
        "provider_calls": generator.calls if generator else 0,
        "prompt_chars": generator.prompt_chars if generator else 0,
    }


def mean(rows, key):
    return statistics.fmean(row[key] for row in rows)


def summarize(rows):
    final_values = [row["final_test_reward"] for row in rows]
    return {
        "runs": len(rows),
        "mean_baseline_search_reward": mean(rows, "baseline_search_reward"),
        "mean_selected_search_reward": mean(rows, "selected_search_reward"),
        "mean_search_gain": mean(rows, "search_gain"),
        "mean_final_test_reward": statistics.fmean(final_values),
        "final_test_stddev": statistics.stdev(final_values) if len(final_values) > 1 else 0.0,
        "improved_search_runs": sum(row["search_gain"] > 0 for row in rows),
        "logical_evaluations": sum(row["logical_evaluations"] for row in rows),
        "provider_calls": sum(row["provider_calls"] for row in rows),
        "prompt_chars": sum(row["prompt_chars"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="examples/benchmark/results.json")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--evaluation-budget", type=int, default=8)
    parser.add_argument("--openai-model")
    args = parser.parse_args()
    if args.runs < 2:
        raise ValueError("runs must be at least 2")
    seeds = (0, 1)
    if args.evaluation_budget < len(seeds) * 2 or args.evaluation_budget % len(seeds):
        raise ValueError("evaluation budget must cover whole paired-seed trials")
    if args.evaluation_budget > 12:
        raise ValueError("evaluation budget exceeds the six-value search space")
    torch.set_num_threads(1)
    random.seed(0)
    np.random.seed(0)

    tasks = [CachedTask(task) for task in (tree_task(), mlp_task(), threshold_task())]
    strategies = ("random", "scripted", "raw_model", "compact_model")
    records = []
    for task in tasks:
        for strategy in strategies:
            rows = [
                one_run(task, strategy, run_index, args.evaluation_budget, seeds, args.openai_model)
                for run_index in range(args.runs)
            ]
            records.append({
                "task": task.name,
                "family": task.family,
                "strategy": strategy,
                "summary": summarize(rows),
                "runs": rows,
            })

    result = {
        "benchmark": "equal-logical-training-budget proposer comparison",
        "provider": args.openai_model or "offline deterministic model fixture",
        "interpretation": (
            "Offline results validate orchestration and context ablation only; "
            "they do not measure LLM intelligence or establish proposer superiority."
            if not args.openai_model
            else "Live provider run; repeat independently before drawing conclusions."
        ),
        "protocol": {
            "runs_per_task_strategy": args.runs,
            "paired_seeds": list(seeds),
            "logical_evaluations_per_run": args.evaluation_budget,
            "final_evaluations_per_run": 1,
            "selection_split": "search",
            "final_split_visible_to_proposer": False,
        },
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
