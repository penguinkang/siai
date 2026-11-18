# Equal-budget proposer benchmark

This harness compares four proposal strategies under the same logical training
budget:

- incumbent-preserving random search;
- a fixed task-independent schedule;
- a model receiving the complete raw journal history; and
- the same model receiving SIAI's compact decision packet.

It runs three independent searches on decision-tree training, PyTorch MLP
training, and classification-threshold tuning. Every search uses paired seeds,
selects only on the search split, and evaluates the selected configuration once
on an isolated final split.

```bash
PYTHONPATH=src .venv/bin/python examples/benchmark/proposer_benchmark.py
```

By default, both model strategies use the same deterministic offline fixture.
That mode validates budgets, selection, reporting, and raw-versus-compact context
plumbing. It is not evidence that an LLM proposer is effective. The recorded
prompt character counts are a context-size measurement, not a token-cost claim.

The committed offline run covers 36 searches: three tasks × four strategies ×
three independent runs. Raw and compact model fixtures produce identical reward
trajectories because they share the same deterministic policy. Compact context
uses 74,538 prompt characters versus 81,696 for raw history, an 8.8% reduction.
Random search leads some toy-task averages, so these results provide no evidence
of proposer superiority.

For a credentialed provider run:

```bash
OPENAI_API_KEY=... PYTHONPATH=src .venv/bin/python \
  examples/benchmark/proposer_benchmark.py --openai-model YOUR_MODEL \
  --output examples/benchmark/live-results.json
```

Live results should be repeated across provider/model versions and independent
search runs before drawing conclusions. Do not commit credentials or unrestricted
final-test access.
