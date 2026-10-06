# RGAT-RoBERTa-RGAT knowledge graph learning

Implementation of the Milestone 3 KG -> LM -> KG methodology, using
**FB15k-237, RGAT, frozen RoBERTa, and ComplEx**. Start with **32-dimensional
KG vectors** for a smaller Colab workload. ComplEx packs 16 real and 16
imaginary coordinates into those 32 values.

The reference is [the current Milestone 3 PDF](docs/Group_03_Milestone_03_2026_07_08.pdf),
Section 3.1 and its architecture diagram (PDF pages 17-21), plus the validation
plan (page 22). The PDF is local and ignored by Git; a fresh clone may not contain it.
Appendix A lists excluded literature sources; it is not an implementation specification.

## Architecture

```text
Random entity embeddings (32) + random relation embeddings (32)
                    |
         Optional KG-only warm-up
                    |
           RGAT encoder phase 1
       entity features + relation features
                    |
        Shared linear projection 32 -> 768
                    |
 [KG soft-prompt token | entity OR relation description tokens]
                    |
          Frozen pretrained roberta-base
           Read the soft-prompt position
                    |
        Shared linear projection 768 -> 32
                    |
        Directly into RGAT encoder phase 2
       final entity + relation representations
                    |
                 ComplEx
                    |
   Plain BCEWithLogits over positive and negative triples
```

There is **no semantic fusion module**, gate, residual blend with encoder-1
outputs, or bridge ramp. The prompt is an additional token with its own
attention-mask entry. Every entity and relation uses this path once warm-up
ends. LM weights stay frozen, but autograd passes through the LM to train the
KG encoders and projections. Encoder-internal graph residuals and LayerNorm
remain part of RGAT; they do not fuse structural and semantic bridge outputs.

Both entity and relation text are aligned to the dataset's exact ID maps.
Each graph stage uses current relation vectors as edge features in RGAT
attention, with opposite signs for inverse message edges. It updates relation
vectors with a learned graph readout of mean oriented endpoint differences.
This relation readout is an explicit implementation choice: the PDF requires
entity/relation representations but does not prescribe RGAT's relation-update
formula. Projected relation representations enter phase 2 and its final relation
vectors reach ComplEx during both training and candidate ranking.

**DistMult is completely removed**: no implementation, configuration, or
comparison option remains. R-GCN is also absent. FB15k-237 and ComplEx-only
are project decisions that override the report's older dataset/scorer plans.
The starting width 32 overrides the report's suggested 256 for the initial
compute budget; 128/200/256 comparisons remain available.

`CLAUDE.md` is preserved as requested previously. Its historical instructions
predate this implementation; this README describes the current executable model.
The `docs/` folder is preserved.

## Training and evaluation

- `proposed`: graph-conditioned entity/relation soft prompts between graph stages.
- `baseline`: the same two graph stages and ComplEx, without an LM.
- `text_enhanced`: unconditioned entity/relation description embeddings initialize
  inputs to the same graph stack. Its prompt is zero; graph features do not
  condition the LM. This is a simple text control, with no fusion module.
- `proposed_no_warmup`: the proposed semantic path is enabled from epoch 1.

The default warm-up is 10 graph-only epochs. The semantic path switches on
fully at epoch 11. Semantic models select their best checkpoint only after that
switch. `warmup_epochs: 0` enables it immediately.

Loss is mean binary cross entropy over concatenated positives and negatives.
There is no self-adversarial weighting or N3 penalty; default weight decay is
zero. Four negatives per positive are balanced between head and tail corruption,
with known training positives excluded. This ordinary example-level mean gives
four negative examples for each positive; it is not separately class-balanced.

Training retains sampled local graphs and a reproducible per-epoch 90%/10%
message-passing/supervision split of **training triples only**. Each epoch
reshuffles that split; an epoch supervises approximately 10% of the training
triples, not a conventional full pass. These are engineering choices beyond
the PDF's high-level methodology and must be disclosed in the experiment report.
Validation/test triples never enter message passing or training-negative filters.
Evaluation uses the full training graph, all entity candidates, filters known
positives from all three splits, and uses average ranks for ties. Metrics are
MRR, Hits@1/3/10, Head_MRR, and Tail_MRR.

## Data and dependencies

```bash
python -m pip install -r requirements.txt
python run_pipeline_smoke_test.py
```

The smoke suite uses a tiny randomly initialized RoBERTa and synthetic triples;
it needs no pretrained downloads and does not establish research performance.

Real runs download standard FB15k-237 triples, entity text, relation text, and
`roberta-base`. Expected counts are 14,541 entities, 237 relations, and
272,115 / 17,535 / 20,466 train/validation/test triples. Incorrect counts fail.
You can supply these files manually and set `download_if_missing: false`:

```text
data/raw/fb15k237/train.txt                 head<TAB>relation<TAB>tail
data/raw/fb15k237/valid.txt
data/raw/fb15k237/test.txt
data/text/fb15k237/FB15k_mid2description.txt entity-key<TAB>description
data/text/fb15k237/entity2text.txt           entity-key<TAB>name or short text
data/text/fb15k237/relation2text.txt         relation-key<TAB>description
```

Entity descriptions prefer the long description, then the short text. An entity
with neither uses an empty description through the same soft-prompt LM path;
coverage is recorded. Missing relation descriptions fail rather than silently
omitting the relation path. The default relation resource contains short labels
from [KG-BERT](https://github.com/yao8839836/kg-bert/tree/master/data/FB15k-237);
replace `relation2text.txt` with reviewed, richer descriptions if desired before
starting experiments. No descriptions are generated from held-out triples.

Data manifests fingerprint entity/relation IDs, splits, and both sets of tokens.
Changing text requires a new experiment; resume rejects changed fingerprints.

## Run in Google Colab

Open [notebooks/colab_training.ipynb](notebooks/colab_training.ipynb) in Colab.
The notebook includes setup, GPU verification, offline tests, a real-data
preflight, a short pilot, persistent checkpoints, resume, and result export.

1. Select **Runtime -> Change runtime type -> GPU**.
2. Upload a ZIP of this updated project using the notebook's first cell. Include
   source, requirements, configs, and tests; exclude `.git`, caches, and old run
   outputs. Alternatively choose its GitHub source option **after these changes
   have been pushed**. Cloning the current remote before then gives the old code.
3. Install requirements and run the offline smoke suite. PyG's basic installation
   works without installing its optional compiled extensions for this model;
   see [PyG installation documentation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
4. Mount Drive for persistent output. Keep working data/model downloads on the
   local runtime, and save checkpoints/configs/results under a dedicated Drive folder.
5. Run the real-data preflight before committing to a training run:

```python
!python run_colab_preflight.py experiments/configs/proposed_complex.yaml
```

This downloads the real resources and executes one full semantic training step,
checking gradients and reporting sampled graph size, step time, and peak CUDA
memory. It is a resource/correctness check, not a benchmark or a guarantee that
the largest training batch/full evaluation fits. Frozen RoBERTa remains the main
compute cost even at KG width 32.

The starting settings are KG width 32, LM micro-batch 4, training batch 8,
2 neighbors per hop, 64 description tokens, mixed precision, and LM activation
checkpointing. If memory is insufficient, reduce `lm_micro_batch` to 1 first,
then training batch size or neighbors. Change settings in a new run and keep
baseline/proposed configurations matched. These settings have not yet been
measured on a Colab GPU.

For a pilot, copy the proposed configuration and set `warmup_epochs: 1`,
`max_epochs: 3`, and `eval_every: 1`. The notebook does this automatically.
This exercises warm-up, semantic training, checkpointing and final evaluation;
it is not a converged research run. Estimate larger-run cost from its epoch times.

For normal training from the project directory:

```python
from google.colab import drive
drive.mount('/content/drive')

!python run_experiment.py experiments/configs/baseline_complex.yaml --output-dir /content/drive/MyDrive/reverse_jaket/baseline32
!python run_experiment.py experiments/configs/proposed_complex.yaml --output-dir /content/drive/MyDrive/reverse_jaket/proposed32
```

Run each training command in its own cell. Outputs are saved each epoch.
After a runtime restart, repeat source/dependency/data setup, mount Drive, and
resume using the **same saved configuration** and latest checkpoint:

```python
!python run_experiment.py /content/drive/MyDrive/reverse_jaket/proposed32/config.json --output-dir /content/drive/MyDrive/reverse_jaket/proposed32 --resume /content/drive/MyDrive/reverse_jaket/proposed32/last_checkpoint.pt
```

The loader accepts JSON as well as YAML. Resume restores optimizer, AMP scaler,
RNG, epoch, history, and best weights; an interrupted epoch restarts from the
last saved epoch. Re-downloads must resolve to the same frozen LM weights and
text tokenization. GPU availability and session duration vary; persistent output
is necessary because runtime files are temporary ([Colab FAQ](https://research.google.com/colaboratory/faq.html)).

## Checkpoints and comparisons

Old gated-fusion checkpoints/configurations are incompatible. Start a new run;
current checkpoint format is 2. A current **matching 32-dimensional baseline**
can supply structural weights using `--warm-start PATH_TO_BEST_CHECKPOINT`.
To skip additional warm-up after that initialization, use a configuration with
`warmup_epochs: 0`. Warm-start is a separate experimental condition; record it.

Every run exports `config.json`, `data_manifest.json`, `environment.json`,
`history.csv`, `last_checkpoint.pt`, `best_checkpoint.pt`, `final_results.json`,
and `run_status.json`. A warm-up checkpoint is saved when warm-up is enabled.
Use `last_checkpoint.pt` to resume; the final best checkpoint is evaluation-only.
Frozen LM weights are omitted, but their identity is recorded and verified.

```bash
python evaluate_checkpoint.py experiments/runs/proposed_complex/best_checkpoint.pt
python run_comparisons.py --seeds 42 --dimensions
python run_comparisons.py --seeds 42 --dimensions --execute
python summarize_results.py experiments/comparisons --output-dir experiments/reports
```

The first comparison command prepares four matched 32-dimensional configs
without training. The second executes them. Without those limiting arguments,
the default plan is 30 runs: four core variants at 32 plus baseline/proposed
pairs at 128/200/256, each with seeds 42/43/44. Scale up only after the pilot.
Use `--output-root /content/drive/MyDrive/reverse_jaket/comparisons` in Colab.
The summarizer exports measured tables, seed statistics and curves; it excludes
synthetic results by default and does not fabricate missing experiments.

## What remains

The requested model paths, BCE training, comparison configurations, resumable
checkpoints and final filtered evaluation are implemented. Remaining work is:

- Run the real-data preflight and pilot on a Colab GPU; measure memory, runtime,
  stability and text coverage. CPU synthetic tests do not validate GPU feasibility.
- Train comparable KG-only, text-enhanced, proposed and no-warm-up experiments.
  Extend to multiple seeds and 128/200/256 only within documented compute limits.
- Review relation labels/entity descriptions for quality; use identical resources
  across comparisons and keep the recorded fingerprints.
- Report final test metrics, curves, resource costs, and whether the semantic
  path improves link prediction. Existing WN18RR/BERT results are a different model.
- Reconcile the report with the authorized FB15k-237, ComplEx-only and initial
  32-dimensional choices. Its ULTRA-pretraining and Wikidata/FewRel/MetaQA claims
  conflict with its from-scratch link-prediction methodology. Those additional
  tasks are not implemented here and are not silently claimed as completed.

The current relation-update formula and sampled training protocol are explicit
implementation choices that should be added to the report for reproducibility.
