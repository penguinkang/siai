import json
import unittest
from siai import Candidate, ModelCard, Observation, ProposalRequest, IntelligentProposer

class ProposerTests(unittest.TestCase):
    def request(self):
        return ProposalRequest(ModelCard('Improve QA', 'accuracy', 'maximize',
            {'learning_rate': 'positive, at most 0.01'}),
            Candidate({'learning_rate': .001, 'batch_size': 8}),
            (Observation('loss', 'training', 'Loss oscillates'),))

    def response(self, **changes):
        return json.dumps(dict(lever='learning_rate', value=.0005,
            hypothesis='Lower rate reduces oscillation', evidence_ids=['loss'],
            expected_effect='Potential accuracy gain', verification='Compare seeds and QA slices', **changes))

    def test_one_call_and_preserved_config(self):
        calls = []
        def generate(prompt):
            calls.append(prompt)
            return self.response()
        p = IntelligentProposer(generate).propose(self.request())
        self.assertEqual(p.candidate(self.request().incumbent).config,
                         {'learning_rate': .0005, 'batch_size': 8})
        self.assertEqual(len(calls), 1)
        self.assertIn('accuracy', calls[0])

    def test_invalid_evidence_lever_and_repeat(self):
        base = json.loads(self.response())
        for changes in ({'evidence_ids': ['invented']}, {'lever': 'evaluator'},
                        {'value': .001}, {'value': float('nan')}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                IntelligentProposer(lambda p: json.dumps(base | changes)).propose(self.request())
