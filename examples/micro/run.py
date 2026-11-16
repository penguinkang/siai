"""Offline CPU micro-projects exercising evidence → proposal → training → test.

The deterministic proposer is a contract fixture, not an intelligent-model eval.
"""
import argparse
import json
from pathlib import Path
import tempfile
import torch
from siai import Candidate, ModelCard, Observation, ProposalRequest, IntelligentProposer


def sklearn_project():
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    x, y = make_moons(n_samples=300, noise=.15, random_state=1)
    xt, xv, yt, yv = train_test_split(x, y, test_size=.4, random_state=2)
    xs, xf, ys, yf = train_test_split(xv, yv, test_size=.5, random_state=3)
    def train(config):
        model = DecisionTreeClassifier(max_depth=config['depth'], random_state=0).fit(xt, yt)
        return model.score(xs, ys), lambda: model.score(xf, yf)
    return train, 'depth', 1, 4


def neural_project(transformer=False):
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(180, 4, 4, generator=generator)
    y = (x[:, :, 0].sum(1) > 0).long()
    def train(config):
        torch.manual_seed(0)
        if transformer:
            class Net(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = torch.nn.TransformerEncoder(
                        torch.nn.TransformerEncoderLayer(4, 2, dim_feedforward=16,
                            dropout=0, batch_first=True), 1, enable_nested_tensor=False)
                    self.head = torch.nn.Linear(4, 2)
                def forward(self, inputs):
                    return self.head(self.encoder(inputs).mean(1))
            model = Net()
        else:
            model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16, 16),
                                        torch.nn.ReLU(), torch.nn.Linear(16, 2))
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        for _ in range(40):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x[:100]), y[:100])
            loss.backward()
            optimizer.step()
        model.eval()
        def score(a, b):
            with torch.no_grad():
                return (model(x[a:b]).argmax(-1) == y[a:b]).float().mean().item()
        return score(100, 140), lambda: score(140, 180)
    return train, 'learning_rate', .00001, .02


def lm_project():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset
    vocab = {'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, 'red': 3, 'blue': 4, 'green': 5}
    raw = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]',
                                       pad_token='[PAD]', eos_token='[EOS]')
    torch.manual_seed(10)
    base = GPT2LMHeadModel(GPT2Config(vocab_size=6, n_positions=32,
        n_embd=16, n_layer=1, n_head=2, bos_token_id=2, eos_token_id=2, pad_token_id=1))
    data = Dataset.from_dict({'text': ['red blue green red blue green'] * 16})
    search = Dataset.from_dict({'text': ['red blue green red blue'] * 2})
    final = Dataset.from_dict({'text': ['blue green red blue green'] * 2})
    with tempfile.TemporaryDirectory() as d:
        # Produce an actual local pretrained checkpoint before adaptation.
        pretrainer = SFTTrainer(model=base, processing_class=tokenizer, train_dataset=data,
            args=SFTConfig(output_dir=d, max_steps=2, max_length=16,
                per_device_train_batch_size=4, use_cpu=True, bf16=False, fp16=False,
                report_to='none', save_strategy='no', disable_tqdm=True))
        pretrainer.train()
    weights = {k: v.detach().clone() for k,v in base.state_dict().items()}
    def train(config):
        model = GPT2LMHeadModel(base.config)
        model.load_state_dict(weights)
        with tempfile.TemporaryDirectory() as d:
            trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=data,
                args=SFTConfig(output_dir=d, max_steps=4, max_length=16,
                    per_device_train_batch_size=4, use_cpu=True, bf16=False, fp16=False,
                    learning_rate=config['learning_rate'], seed=0,
                    report_to='none', save_strategy='no', disable_tqdm=True))
            trainer.train()
            reward = -trainer.evaluate(eval_dataset=search)['eval_loss']
            def score():
                return -trainer.evaluate(eval_dataset=final)['eval_loss']
            # Trainer remains usable in memory after its temporary output is removed.
            return reward, score
    return train, 'learning_rate', .00001, .003


def run(name, project):
    train, lever, initial, changed = project()
    incumbent = Candidate({lever: initial})
    before, _ = train(incumbent.config)
    request = ProposalRequest(
        ModelCard('Improve search reward', 'accuracy' if name != 'lm' else 'negative_loss',
                  'maximize', {lever: 'positive numeric value'}), incumbent,
        (Observation('baseline', 'task', 'Measured baseline search reward', before),))
    calls = []
    def generate(prompt):
        calls.append(prompt)
        return json.dumps(dict(lever=lever, value=changed,
            hypothesis='Test a larger capacity or update rate', evidence_ids=['baseline'],
            expected_effect='Potential reward increase', verification='Compare fixed search split'))
    proposal = IntelligentProposer(generate).propose(request)
    candidate = proposal.candidate(incumbent)
    after, candidate_test = train(candidate.config)
    accepted = after > before
    # Refit selected baseline only if rejected; never select using test results.
    test_score = candidate_test() if accepted else train(incumbent.config)[1]()
    assert len(calls) == 1
    return dict(project=name, baseline=before, proposed=after, accepted=accepted,
                final_test=test_score, proposal=dict(lever=proposal.lever, value=proposal.value, hypothesis=proposal.hypothesis,
                    evidence_ids=proposal.evidence_ids, expected_effect=proposal.expected_effect,
                    verification=proposal.verification),
                proposer='deterministic contract fixture', seed=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='micro-results.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = [run(name, project) for name, project in (
        ('sklearn', sklearn_project), ('dl', neural_project),
        ('transformer', lambda: neural_project(True)), ('lm', lm_project))]
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

if __name__ == '__main__': main()
