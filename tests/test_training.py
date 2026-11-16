import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from siai.training import SFTAdapter, sft_factory

class TrainingTests(unittest.TestCase):
    def test_shared_contract(self):
        for backend in ('trl', 'unsloth'):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as d:
                calls = []
                class Config:
                    def __init__(self, **kw): self.kw = kw
                class Trainer:
                    def __init__(self, **kw):
                        calls.append(kw)
                        self.state = SimpleNamespace(log_history=[{'loss': 1., 'step': 1}])
                    def train(self): pass
                    def save_model(self, path): Path(path).mkdir()
                class Fast:
                    @staticmethod
                    def from_pretrained(**kw): return 'fresh-model', 'tokenizer'
                    @staticmethod
                    def get_peft_model(model, **kw): return model
                modules = {'trl': SimpleNamespace(SFTConfig=Config, SFTTrainer=Trainer),
                    'transformers': SimpleNamespace(set_seed=lambda s: None),
                    'unsloth': SimpleNamespace(FastLanguageModel=Fast)}
                with patch.dict('sys.modules', modules):
                    adapter = SFTAdapter(sft_factory(backend, 'model', []), lambda t: .7)
                    result = adapter.train({'learning_rate': .001}, seed=1, output_dir=Path(d)/'trial')
                    self.assertEqual(result.reward, .7)
                    self.assertTrue(Path(result.checkpoint).exists())
                    self.assertEqual(calls[0]['args'].kw['learning_rate'], .001)
                    self.assertIn('log-0-loss', [o.id for o in result.observations])
                    with self.assertRaises(FileExistsError):
                        adapter.train({}, seed=1, output_dir=Path(d)/'trial')
    def test_invalid_configuration_before_training(self):
        def fail(*a): self.fail('factory should not run')
        for config in ({'learning_rate': -1}, {'warmup_ratio': 2}, {'evaluator': 1}):
            with self.assertRaises(ValueError):
                SFTAdapter(fail, fail).train(config, seed=0, output_dir='unused')
