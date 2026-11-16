import tempfile
import unittest
from pathlib import Path
from siai import Candidate, Experiment

class Experiments(unittest.TestCase):
    def test_selection_budget_and_lineage(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'run.jsonl'
            result = Experiment(lambda c,s,t: c['x'],
                lambda b,h: Candidate({'x': -1}), budget=6, journal=p).run(Candidate({'x': 1}))
            self.assertEqual(result.best.config, {'x': 1})
            self.assertEqual(result.evaluations, 6)
            self.assertEqual(len(p.read_text().splitlines()), 2)
            with self.assertRaises(FileExistsError):
                Experiment(None, None, journal=p).run(Candidate({}))

    def test_failed_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                Experiment(lambda *a: float('nan'), None, journal=Path(d)/'r').run(Candidate({}))

    def test_duplicate_stops(self):
        with tempfile.TemporaryDirectory() as d:
            r = Experiment(lambda *a: 1, lambda b,h: b, journal=Path(d)/'r').run(Candidate({}))
            self.assertEqual(r.evaluations, 3)

if __name__ == '__main__': unittest.main()
