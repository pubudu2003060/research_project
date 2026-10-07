# Semantic training performance and AMP diagnostics

Inspected base commit: `3ab50df3d804fb86a393a9b345f18c27ae83833f`.
Local verification: 2026-10-07, Windows, PyTorch 2.11.0+cpu, no CUDA device.
No FB15k-237/RoBERTa-base GPU timing or GPU stability result is claimed here.

## Findings and correctness

1. The original local forward sends every relation through the frozen LM.
   With 237 relations and a micro-batch of 4 this is 60 relation LM calls per
   training batch, or 204,120 per 3,402-batch semantic epoch, before backward
   checkpoint recomputation and entity processing. Frozen weights still require
   input-gradient computation, so freezing does not eliminate backward cost.
2. Only relations appearing in sampled message edges OR the scored positive/
   negative triples affect a local batch. The encoder's relation readout and
   LayerNorm operate per row; transformer examples do not attend across rows.
   The optimized path takes this union (inverse edges use their canonical
   relation ID), projects those CURRENT relation vectors through the LM, and
   scatters the results into the global-ID table for RGAT/scoring. Omitted rows
   are marked unavailable; scoring them raises an error. Evaluation and calls
   without explicit scored relation IDs retain the complete relation table.
   This is elimination of unused computation, not relation subsampling.
3. Token IDs/masks are copied to the model device once and invalidated when
   source tensors change. No LM output or trainable representation is cached.
   Text transformer states depend on the current soft prompt even though the
   transformer weights are frozen. Caching those states would change the model.
   Word lookup embeddings alone are invariant but cheap relative to transformer
   processing; caching the complete entity embedding tensor would consume much
   more GPU memory, so it is not enabled without a measured benefit.
4. The old preflight's single error conflated None, all-zero, NaN and Inf
   gradients. It checked before GradScaler could skip/back off on overflow.
   The new report distinguishes these cases. For non-finite gradients with a
   finite forward loss, GradScaler skips the update and backs off; the exact
   batch and dropout RNG are retried, at most `amp_max_retries` (default 16).
   Finite successful gradients are still checked in every required module by
   the preflight. Frozen LM parameters must have no gradients. Non-finite
   forward loss, persistent overflow and disconnected paths still fail.
   The precise source of the reported Colab failure remains unverified until
   its real GPU diagnostics are collected. CPU tests reproduce loss-scaling
   overflow and validate recovery; they cannot identify a Colab CUDA kernel.
   In the full two-RGAT/tiny-RoBERTa CPU FP16 reproduction, initial scale 65,536
   produced NaNs in all 12 encoder1 gradient tensors (none were missing), plus
   NaN/Inf gradients downstream. Three retries reduced the scale to 8,192;
   every required module then had finite nonzero gradients. No optimizer update
   occurred for an invalid attempt. The diagnostic is saved in
   `experiments/reports/performance/cpu_amp_diagnostic.json`.
5. `amp_dtype` supports `float16` (backward-compatible default) and `bfloat16`
   on supporting CUDA GPUs. BF16 uses no loss scaler. CPU research execution
   remains FP32. No automatic precision changes or silent FP32 fallback occur.
6. Checkpointing re-executes LM forwards during backward. KG dimension 32 does
   not shrink RoBERTa's hidden width 768 or its transformer activations.
   The default remains on, and relation micro-batch size defaults to the entity
   micro-batch size. Selecting new defaults requires GPU time/memory profiles.
7. `2 * layers_per_phase` covers the sequential entity-message paths of the two
   RGAT stages. The relation readout also couples endpoints across the sampled
   graph, so this is not a proof of equivalence to full-graph training. Do not
   remove boundary entities: they contribute to relation readouts too. The
   existing sampled training protocol, graph size, and hop count are unchanged.

## Local measurements (synthetic only)

Command:

```bash
python profile_semantic.py --synthetic --full-epoch --cases legacy optimized no_checkpoint larger_relation_batch --output experiments/reports/performance/cpu_synthetic_profile.json
```

This uses 128 entities, 237 relations, 512 synthetic training triples and a
random one-layer LM with hidden width 32, **not pretrained RoBERTa-base**.
Other initial settings include KG width 32, 64 tokens, mp_ratio 0.90 and batch
size 8. The complete synthetic epoch has seven training batches. Each case
starts from identical trainable weights and sampling/dropout seeds.

| Measurement | Original relation computation, checkpoint on | Optimized, checkpoint on | Optimized, checkpoint off | Optimized, relation micro-batch 16, checkpoint on |
|---|---:|---:|---:|---:|
| First batch including diagnostics (s) | 1.071 | 0.743 | 0.429 | 0.696 |
| Mean of batches 2–7 (s) | 1.070 | 0.496 | 0.462 | 0.627 |
| Measured seven-batch training time (s) | 7.492 | 3.718 | 3.203 | 4.461 |
| Measured epoch including setup/validation (s) | 8.079 | 3.908 | 3.606 | 4.537 |
| Estimated training epoch from steady mean (s) | 7.490 | 3.470 | 3.235 | 4.392 |
| LM calls including recomputation, training only | 1196 | 736 | 368 | 456 |
| First-batch relation prompts | 237 | 109 | 109 | 109 |
| GPU allocated/peak memory | unavailable | unavailable | unavailable | unavailable |

Raw stages are in `experiments/reports/performance/cpu_synthetic_profile.json`.
That report is ignored by Git along with other generated reports. These are
single harness measurements, affected by CPU load and cold-start overhead;
they are not publication benchmarks or evidence of a GPU speedup. No FB15k-237
epoch time is extrapolated from this tiny LM. Larger batches were not uniformly
faster here, which is another reason not to choose GPU defaults from CPU data.

## Colab: diagnose the actual failure

Use the updated source, install requirements, mount Drive, and run from the
project directory. Use a fresh output filename per invocation. For example:

```python
from pathlib import Path
DIAG = Path('/content/drive/MyDrive/reverse_jaket/diagnostics')
DIAG.mkdir(parents=True, exist_ok=True)
```

Reproduce the first-step FP16 gradient classification with the old relation
computation and no retries (a failure here is intentionally diagnostic):

```python
!python run_colab_preflight.py experiments/configs/proposed_complex.yaml --legacy --amp-dtype float16 --amp-max-retries 0 --output "{DIAG}/legacy_fp16_first_attempt.json"
```

Then validate normal recovery and the optimized path:

```python
!python run_colab_preflight.py experiments/configs/proposed_complex.yaml --amp-dtype float16 --output "{DIAG}/optimized_fp16.json"
```

On a BF16-capable GPU, compare explicitly:

```python
import torch
if torch.cuda.is_bf16_supported():
    !python run_colab_preflight.py experiments/configs/proposed_complex.yaml --amp-dtype bfloat16 --output "{DIAG}/optimized_bf16.json"
```

The JSON reports unscaled gradient categories, finite magnitudes, scale changes,
overflow retries, actual precision, GPU name, prompt counts, stages and memory.
Failures are saved as well as raised; they do not become a false "passed".

## Colab: matched performance measurements

Start with five batches. Use `--amp-dtype float32` to match the reported FP32
pilot, then profile FP16 separately after its preflight succeeds:

```python
!python profile_semantic.py experiments/configs/proposed_complex.yaml --batches 5 --amp-dtype float32 --cases legacy optimized no_checkpoint larger_relation_batch --relation-micro-batch 16 --output "{DIAG}/semantic_fp32_batches.json"
```

For realistic post-warm-up prompts, pass the pilot's saved configuration and
warm-up checkpoint instead of the repository configuration:

```python
PILOT = '/content/drive/MyDrive/reverse_jaket/pilot32'
!python profile_semantic.py "{PILOT}/config.json" --checkpoint "{PILOT}/warmup_checkpoint.pt" --batches 5 --cases legacy optimized no_checkpoint larger_relation_batch --output "{DIAG}/pilot_semantic_batches.json"
```

Each profile case restores the same model weights and uses a fresh optimizer,
so these are controlled performance runs, not continuation of the pilot.
The loaded checkpoint's frozen LM, dataset/text identity and architecture are
verified. Optimizer state is deliberately not restored for these matched cases.

After reviewing step times and memory, measure complete semantic epochs:

```python
!python profile_semantic.py "{PILOT}/config.json" --checkpoint "{PILOT}/warmup_checkpoint.pt" --full-epoch --cases legacy optimized --output "{DIAG}/pilot_semantic_epochs.json"
```

This runs **two full semantic training epochs, each followed by full validation**;
it can take hours. It does not overwrite the pilot or produce research results.
Reports are saved after each case. Short profiles include an estimated training
epoch time derived from their own measured non-first batches; only `--full-epoch`
reports a measured full epoch. Estimates exclude validation and checkpoint I/O.
Stages cover sampling, both phase-1 paths, both semantic bridges, encoder2,
scoring/loss, backward, gradient checks/clipping and optimizer step. CUDA events
are resolved per batch; Python wall time and GPU stage time need not sum exactly.
Training and validation peaks are recorded separately. An OOM is recorded as
a failed case, not hidden or retried with different research settings.

Display the profile:

```python
import json
import pandas as pd
report = json.loads((DIAG / 'pilot_semantic_batches.json').read_text())
display(pd.DataFrame([
    {'case': name, 'status': r['status'],
     'first_batch_s': r.get('cold_batch_seconds'),
     'steady_batch_s': r.get('steady_batch_mean_seconds'),
     'estimated_training_epoch_s': r.get('estimated_training_epoch_seconds'),
     'peak_GPU_MiB': r.get('training_memory', {}).get('cuda_peak_mib')}
    for name, r in report['cases'].items()
]))
```

## Resume and research controls

No parameter names/shapes or checkpoint format changed. Existing v2 configs
receive defaults for new execution options, and same-config resume restores
optimizer/scaler/RNG/history. Do not change `use_amp`, `amp_dtype`, batching or
other experiment settings while resuming; start a separately recorded run to
compare precision or checkpointing settings. Logging/profiling options may
change on resume. A resumed old run uses the new equivalent computation and
records `execution_implementation` in environment.json; it is not claimed to
be bitwise identical across software versions. Seed statistics distinguish
execution/precision settings. The comparison runner preserves completed runs.

Research invariants remain FB15k-237, ComplEx, two RGAT stages, trainable entity
and relation prompts through a frozen LM, filtered evaluation, the original
negative sampler, mp_ratio 0.90, batch size 8, four negatives, width 32, and the
original warm-up/epoch schedules.

## Tests and changed files

`python run_pipeline_smoke_test.py`: 27 tests discovered, 26 passed, one CUDA-only
test skipped locally. Existing 16 tests all pass, including exact uninterrupted
versus resumed parameter equality. New tests cover original-path output/gradient
equivalence, inverse and score-only relations, prompt dependence on current KG
weights, cache invalidation, gradient categories, real CPU FP16 scaler overflow
recovery, bounded failure, CPU FP16/BF16 graph/LM gradients, structural initialization,
checkpoint recomputation, old-v2 configuration loading and progress/profiling.

| Files | Change |
|---|---|
| `models/reverse_jaket.py` | Active relation union, guarded partial representations, token cache, model-stage timers |
| `models/kg_lm_bridge.py` | Bulk token transfer before micro-batch loop |
| `models/frozen_lm.py` | Count forward invocations including recomputation |
| `training/batches.py` | Shared unchanged negative sampling/local sampling/loss construction |
| `training/precision.py` | Explicit precision selection, gradient reports, bounded checked scaler steps |
| `training/profiling.py` | Opt-in CUDA-event/CPU timers and memory readings |
| `training/config.py` | Execution, precision, logging and profiling options |
| `training/train_pipeline.py` | Shared step, compact progress, profiles, environment fields, v2 defaults |
| `run_colab_preflight.py` | Checked semantic step and saved diagnostic reports |
| `profile_semantic.py` | Controlled before/after and checkpointing batch/epoch benchmark harness |
| `run_comparisons.py` | Normalize legacy saved configs when checking completed runs |
| `summarize_results.py` | Keep precision/execution settings separate in statistical groups |
| `tests/test_pipeline.py` | Regression coverage listed above |
| `notebooks/colab_training.ipynb` | Correct repository URL and optional GPU diagnostics/profiling cells |
| `README.md`, `PERFORMANCE_NOTES.md` | Behavior, measurements, commands and limitations |

PyTorch's expected AMP behavior:
[AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).
Checkpoint recomputation semantics:
[checkpoint documentation](https://docs.pytorch.org/docs/stable/checkpoint.html).
