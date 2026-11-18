# v0.1.0 proof of concept: TRL SFTTrainer

Decision (November 16, 2026): use Hugging Face TRL's SFTTrainer as the
first integration target, with Unsloth as the acceleration path. Both now use
the shared `SFTAdapter` contract and have completed real end-to-end training
runs on the development examples branch.

## Why this target

[SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) exposes training and
evaluation datasets, custom metrics, callbacks and PEFT configuration. These
interfaces fit our existing proposer contract without modifying trainer internals.
Its documented Unsloth integration also leaves a path to accelerated training:
[Unsloth fine-tuning guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide).
TRL and Unsloth need not be mutually exclusive. Start with one training path to
keep failures attributable. Select and lock dependency versions when the adapter
is implemented and tested; do not claim compatibility from documentation alone.

## Minimal experiment

Use a small causal language model on a fixed, programmatically scored task such
as structured extraction. Freeze model revision, tokenizer, training dataset,
search-validation dataset, final-test dataset, decoding and scoring implementation.
Start every candidate from the same base checkpoint and adapter initialization
policy with paired seeds. The first PoC optimizes the training recipe; it does
not repeatedly continue training the incumbent weights.

Model-card reward: exact field-match accuracy on search-validation examples.
Report invalid-output rate, per-field accuracy and training cost separately.
Training loss is diagnostic feedback, not a substitute for task reward.
Final test remains unavailable to the proposer.

Initially expose only learning_rate and warmup_ratio, with machine-validated
bounds. Keep batch size, sequence length, examples seen and optimizer updates
fixed. Set a hard trial count and record actual tokens and elapsed time, since
fixed steps alone do not establish equal compute.

## Adapter responsibilities

1. Validate the proposed lever and value before training.
2. Create a fresh model and SFTConfig, seed all training randomness and train.
3. Evaluate generated outputs with the fixed task scorer.
4. Convert log history and task results into named observations: loss trajectory,
   gradient norms when available, validation reward slices, malformed outputs,
   runtime and training errors. Preserve raw artifacts and configuration versions.
5. Build ProposalRequest with the model card, incumbent, observations, attempted
   candidates and remaining budget. Persist the complete returned Proposal.
6. Train the proposal, compare paired rewards and retain or reject the candidate.
   Apply the model-card direction and require every metric guardrail to pass.

Custom reward scoring happens outside SFTTrainer's training loss. A dedicated
adapter must bridge these observations into the intelligent proposer; the current
scalar Experiment callback alone does not do so.

## Acceptance criteria

A runnable baseline → feedback → proposal → fresh training → reward comparison
loop; rejection of invalid values before GPU work; complete records for failures;
and a replayable fixed-seed configuration. Compare intelligent proposals with
random and fixed scripted search under the same training and proposal budgets.
Report reward versus total cost over multiple independent searches, including
negative results. Effectiveness is an empirical milestone, not guaranteed by a
successful smoke test. GPU-specific validation and a live credentialed provider
call remain outstanding.

Update: both backend factories and a shared `SFTAdapter` are implemented in
`siai.training`. The development branch records a real TRL run and a real
Unsloth MLX run, including proposal transport, training, selection, checkpoint
reload, and isolated final scoring. These are compatibility smoke tests, not an
equal-compute comparison or evidence of proposer superiority. Exact versions are
recorded beside each measured result on `dev/micro-projects`.
