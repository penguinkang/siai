import json
import unittest
from dataclasses import dataclass
from siai import (Candidate, MetricGuardrail, ModelCard, Observation, ProposalRequest,
                  IntelligentProposer, build_proposal_context)

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

    def test_provider_metadata_is_attached_without_changing_model_contract(self):
        @dataclass
        class Metadata:
            provider: str
            model: str

        class Generate:
            last_metadata = Metadata("test", "model")

            def __call__(self, prompt):
                return self_outer.response()

        self_outer = self
        proposal = IntelligentProposer(Generate()).propose(self.request())
        self.assertEqual(proposal.provider, {"model": "model", "provider": "test"})

    def test_context_ranks_and_caps_evidence(self):
        request = ProposalRequest(
            self.request().card,
            self.request().incumbent,
            (
                Observation('runtime', 'cost', 'Runtime seconds', 2.0),
                Observation('accuracy', 'metric', 'QA accuracy regressed', .4),
                Observation('oom', 'failure', 'Out of memory'),
            ),
        )
        context = build_proposal_context(request, max_observations=2)
        self.assertEqual([row['id'] for row in context['observations']], ['oom', 'accuracy'])
        self.assertEqual(context['observation_count'], 3)
        self.assertEqual(context['observations_omitted'], 1)

    def test_context_summarizes_directional_experiment_gain(self):
        card = ModelCard('Lower loss', 'loss', 'minimize', {'learning_rate': 'positive'})
        history = (
            {'type': 'trial', 'candidate': 'base', 'parent': None, 'mean': 2.0,
             'accepted': True, 'errors': [], 'proposal': None},
            {'type': 'trial', 'candidate': 'next', 'parent': 'base', 'mean': 1.5,
             'accepted': True, 'errors': [],
             'proposal': {'lever': 'learning_rate', 'value': .01, 'hypothesis': 'faster'}},
        )
        request = ProposalRequest(card, Candidate({'learning_rate': .01}),
                                  (Observation('loss', 'metric', 'Loss', 1.5),), history)
        context = build_proposal_context(request)
        self.assertEqual(context['experiments'][1]['directional_gain'], .5)
        self.assertEqual(context['experiments'][1]['lever'], 'learning_rate')

    def test_prompt_uses_decision_packet(self):
        captured = {}
        def generate(prompt):
            captured['context'] = json.loads(prompt.split('REQUEST:\n', 1)[1])
            return self.response()
        IntelligentProposer(generate, max_observations=1).propose(self.request())
        self.assertIn('experiments', captured['context'])
        self.assertNotIn('history', captured['context'])

    def test_invalid_observation_limit(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                IntelligentProposer(lambda _: self.response(), max_observations=value)

    def test_model_card_validates_guardrails(self):
        card = ModelCard(
            'Improve QA', 'accuracy', 'maximize', {'learning_rate': 'positive'},
            guardrails=(MetricGuardrail('latency_ms', 'at_most', 100),),
        )
        self.assertTrue(card.guardrails[0].passes(99))
        self.assertFalse(card.guardrails[0].passes(101))
        with self.assertRaises(ValueError):
            MetricGuardrail('latency_ms', 'lower', 100)
        with self.assertRaises(ValueError):
            ModelCard(
                'Improve QA', 'accuracy', 'maximize', {'learning_rate': 'positive'},
                guardrails=(
                    MetricGuardrail('latency_ms', 'at_most', 100),
                    MetricGuardrail('latency_ms', 'at_least', 1),
                ),
            )
