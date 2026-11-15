# Compositional Failures in CLIP: Improving Attribute-Object Binding via Hard-Negative Fine-Tuning

Foundations of Generative AI, lab project. Team: **TODO names**. All experiments ran on a free Google Colab T4.

## Summary of findings

We fine-tuned CLIP ViT-B/32 with LoRA on COCO captions, with and without structured hard negatives, and evaluated on
Winoground (primary), SugarCrepe (secondary) and COCO retrieval. All numbers below come from `results/summary.md`.

1. **Winoground: no improvement on any metric.** The version trained on all validated negatives (C) has a *lower* text score than the
   baseline (B): -4.5 points, 95% CI [-7.75, -1.33], Holm-adjusted p = 0.040. Image and group scores show no detectable difference for any condition.
2. **SugarCrepe: gains on two task families that survive multiple-comparison correction.** C beats B on `replace_rel` (+6.2 points, Holm p = 0.002)
   and `add_att` (+3.7, Holm p = 0.014). The `swap_att` and `replace_obj` differences do not survive correction.
3. **The in-distribution gains look like text-only cues.** On our own validation negatives, C's accuracy on `obj_swap` (93.0%) and `rel_replace` (95.5%)
   equals that of a text-only classifier (93.0%, 96.0%), and Winoground, whose two captions per image are equally plausible, does not improve.
4. **The Winoground text drop comes from relation-category negatives.** Training on `rel_replace` + `obj_swap` negatives alone reproduces it in all three seeds
   (29.2 +/- 0.3 text, vs 33.9 +/- 1.4 for B); attribute-only and object-only runs (one seed each) look like B.
5. **Cost:** about 2 to 3 points of image-to-text R@1 on held-out COCO images.

We make no claim that structured hard negatives improve compositional understanding in general. Evidence for attribute-object binding specifically is
essentially absent (see Limitations).

## Design

| Run | Tuning | Objective | Purpose |
|---|---|---|---|
| A `zeroshot` | none | none | pretrained CLIP ViT-B/32 reference |
| B `lora_baseline` | LoRA | L_CLIP (lambda = 0) | effect of fine-tuning alone |
| C `lora_hn` | same LoRA | L_CLIP + lambda * L_HN, all validated negatives | effect of hard negatives (B vs C) |
| C-plaus `lora_hn_plaus` | same LoRA | same, likelihood-matched negatives only | mitigates text-only shortcuts |

B and C use identical LoRA adapters (r = 16, alpha = 32, dropout 0.05 on `q_proj`/`v_proj` of both towers; 983,040 trainable parameters, 0.65%),
the same images, the same caption per image per epoch and the same seeds; only the objective differs. Training: lr 1e-4, batch 128, 10 epochs,
weight decay 0.01, 20 warm-up steps, fp16, **lambda fixed at 1.0 in advance** (it was not selected on any data).
L_CLIP is symmetric InfoNCE over in-batch positives. L_HN = -log sigmoid(scale * (s_pos - s_neg)) for each caption that has a negative, on the same image.
B and C each ran with seeds 0, 1, 2; zero-shot is a single deterministic evaluation (its Winoground text score varied by one example, 31.25 vs 31.0, across sessions).

## Data and leakage prevention

- Images: a random subset of **COCO train2017**, 8,000 images split by **image ID** into 6,500 train / 500 validation / 1,000 test; splits are disjoint (asserted in
  `scripts/prepare_coco_subset.py`, recorded in `data/splits.json`). Train uses 3 captions per image (19,500 captions); validation 2 per image; test all 5.
- COCO **val2017 is never used** for training or tuning because SugarCrepe is built on it.
- Winoground and SugarCrepe were never used to generate negatives, choose lambda, epochs or checkpoints (final-epoch models are evaluated; Winoground is logged during training for monitoring only).
- The retrieval test split comes from the same distribution as training, so retrieval gains partly reflect domain adaptation.

## Hard negatives

Rule-based (spaCy), five candidate types: `attr_replace` (red car to green car), `obj_replace` (dog to cat), `rel_replace` (on to under),
`obj_swap` (subject/object swap) and `attr_swap` (attribute-object binding swap, same attribute category only). Each candidate passes a validation stage
(identical, not a permutation, symmetric relation, same-noun swap, synonym replacement, agreement mismatch, complexity).

| Type | Train captions with a validated negative | After likelihood matching (tau = 0.2) |
|---|---|---|
| obj_replace | 10,162 (52.1%) | 5,500 (54.1% kept) |
| rel_replace | 7,870 (40.4%) | 999 (12.7%) |
| obj_swap | 7,689 (39.4%) | 594 (7.7%) |
| attr_replace | 5,718 (29.3%) | 3,442 (60.2%) |
| attr_swap | 251 (1.3%) | 189 (75.3%) |
| any type | 15,922 of 19,500 (81.7%) | 8,638 (44.3%) |

**Shortcut analysis.** A negative is a shortcut risk if it can be told from its positive without the image. Before filtering, GPT-2 likelihood ranks the
negative below its positive 97.8% of the time for `obj_swap` and 95.4% for `rel_replace`; a unigram+bigram classifier trained on train pairs and tested on
validation pairs reaches 93.0% and 96.0%. The word "under" appears in 6,753 `rel_replace` negatives and 221 positives. Likelihood matching reduces this
(GPT-2: 70.6% and 80.6%) but does not remove it (learned detector after filtering: 95.0% on 20 `obj_swap` pairs and 91.1% on 45 `rel_replace` pairs;
these validation counts are small).

**Manual audit** (`scripts/audit_negatives.py`, 40 negatives per type, label 1 = clearly false caption): **TODO precision per type with 95% CI.**

## Evaluation and statistics

Winoground text/image/group (400 examples), SugarCrepe (7,511 examples, 7 tasks; COCO val2017 images), COCO test retrieval (1,000 images, 5,000 captions),
and a hard-negative probe on validation pairs. Mean +/- sample std over seeds. Paired bootstrap over examples (10,000 resamples, differences averaged over
seeds). A difference is called real only if the 95% CI excludes 0, the difference exceeds the seed spread, and every test seed lies beyond the reference's seed range.
Holm-adjusted p-values are reported for the pre-planned comparisons (B vs C, B vs C-plaus, C-plaus vs C: 9 Winoground and 21 SugarCrepe tests); ablation rows and the per-tag
table are exploratory and uncorrected.

## Results

### Winoground and retrieval (mean +/- std over 3 seeds; ablations other than relation-only: 1 seed)

| Condition | Text | Image | Group | i2t R@1 | t2i R@1 |
|---|---|---|---|---|---|
| A zero-shot | 31.0 | 11.2 | 9.0 | 68.9 | 49.0 |
| B baseline | 33.9 +/- 1.4 | 12.3 +/- 1.8 | 8.7 +/- 0.8 | 75.0 +/- 0.9 | 60.0 +/- 0.2 |
| C all negatives | 29.4 +/- 0.1 | 11.9 +/- 0.5 | 8.0 +/- 0.4 | 73.1 +/- 0.2 | 59.7 +/- 0.2 |
| C-plaus matched | 33.3 +/- 1.0 | 12.0 +/- 0.2 | 8.8 +/- 0.4 | 72.8 +/- 0.3 | 59.5 +/- 0.4 |
| relation-only (3 seeds) | 29.2 +/- 0.3 | 12.3 +/- 0.4 | 7.8 +/- 0.2 | 71.9 +/- 0.3 | 59.7 +/- 0.1 |
| attribute-only | 33.2 | 11.2 | 7.5 | 73.2 | 59.3 |
| object-only | 34.0 | 11.2 | 8.0 | 73.3 | 59.5 |
| lambda 0.5 | 30.8 | 12.8 | 8.0 | 73.9 | 60.1 |
| lambda 2 | 28.2 | 11.8 | 8.2 | 71.5 | 59.5 |

Winoground chance: 25 / 25 / 16.7. Tag breakdown: C's text drop is concentrated in the Object tag (141 examples): -9.0 points vs B,
CI [-15.4, -2.8] (exploratory); the Relation tag shows no detectable difference; the Both tag (26 examples) is not interpretable.

### SugarCrepe accuracy (%)

| Condition | replace_obj | replace_att | replace_rel | swap_obj | swap_att | add_obj | add_att | mean |
|---|---|---|---|---|---|---|---|---|
| A zero-shot | 90.7 | 80.3 | 69.4 | 61.2 | 64.0 | 77.1 | 69.5 | 73.2 |
| B baseline | 94.2 +/- 0.2 | 85.8 +/- 1.2 | 69.3 +/- 0.2 | 63.1 +/- 0.6 | 69.2 +/- 0.8 | 86.5 +/- 0.4 | 78.0 +/- 0.6 | 78.0 +/- 0.2 |
| C all negatives | 93.4 +/- 0.3 | 86.3 +/- 0.8 | 75.5 +/- 0.3 | 68.2 +/- 1.1 | 72.1 +/- 0.9 | 85.6 +/- 0.3 | 81.7 +/- 1.0 | 80.4 +/- 0.2 |
| C-plaus matched | 93.6 +/- 0.2 | 86.8 +/- 0.1 | 73.6 +/- 0.5 | 66.5 +/- 1.1 | 71.2 +/- 0.5 | 86.8 +/- 0.2 | 82.1 +/- 1.1 | 80.1 +/- 0.1 |
| relation-only (3 seeds) | 92.9 +/- 0.1 | 85.1 +/- 0.3 | 75.6 +/- 0.6 | 68.6 +/- 0.4 | 71.1 +/- 0.6 | 83.5 +/- 0.1 | 80.2 +/- 1.1 | 79.6 +/- 0.1 |
| attribute-only | 93.5 | 86.7 | 70.3 | 66.5 | 71.6 | 85.4 | 82.1 | 79.4 |
| object-only | 93.7 | 84.0 | 69.7 | 63.7 | 67.4 | 88.5 | 79.0 | 78.0 |

Compare against B, not zero-shot: plain COCO fine-tuning already raises most tasks because SugarCrepe is COCO-derived. Pre-planned comparisons that survive
Holm correction: C vs B on `replace_rel` (+6.24, p = 0.002) and `add_att` (+3.66, p = 0.014); C-plaus vs B on `replace_rel` (+4.27, p = 0.002) and `add_att`
(+4.09, p = 0.002); C vs C-plaus on `replace_rel` (+1.97, p = 0.022) and `add_obj` (-1.20, p = 0.020, C lower). Exploratory ablations suggest each negative
category helps the SugarCrepe family of the matching category (relation-only: `replace_rel` +6.3 and `swap_obj` +5.4 vs B; attribute-only: `add_att` +4.8; object-only: `add_obj` +1.5;
single seeds except relation-only).

### Hard-negative probe (validation pairs, caption ranked above its negative, %)

| Type (pairs) | Zero-shot | B | C | C-plaus |
|---|---|---|---|---|
| attr_replace (289) | 77.5 | 81.5 | 90.1 | 88.1 |
| attr_swap (12, too few to interpret) | 66.7 | 61.1 | 72.2 | 75.0 |
| obj_replace (542) | 89.1 | 93.8 | 95.8 | 95.4 |
| obj_swap (412) | 71.4 | 59.5 | 93.0 | 80.2 |
| rel_replace (425) | 60.9 | 61.7 | 95.5 | 91.9 |

Figures: `figures/winoground.png`, `sugarcrepe.png`, `retrieval.png`, `ablation.png`, `probe.png`, `loss_curves.png`.
Qualitative Winoground examples: **TODO** (`figures/qualitative_*.png`).

## What these results do and do not support

- Supported: relation and attribute hard negatives improve SugarCrepe `replace_rel` and `add_att`; unfiltered relation-type negatives lower Winoground text; nothing improves Winoground group or image.
- Consistent with, not tested: the mechanism is text-only cue learning (the probe matches a text-only detector; the matched-negatives variant, which learned less of the cue, did not hurt Winoground text).
  The Object-tag drop is concentrated where Winoground pairs differ by swapping objects, the same operation as `obj_swap`; we did not test why.
- Not supported: any claim about attribute-object binding. `attr_swap` has 251 training captions (189 after matching) and 12 validation pairs, and its SugarCrepe `swap_att` gain does not survive correction.

## Limitations and deviations

- One model, one LoRA configuration, no hyperparameter search; lambda 0.5 and 2 are single-seed sensitivity runs.
- Winoground has 400 examples; ablations other than relation-only have one seed; per-tag and ablation tests are exploratory; retrieval has no per-example data, so no test.
- GPT-2 likelihood is a crude proxy for plausibility; the likelihood filter reduces but does not remove text-only separability.
- Rule-based negatives only; no LLM-written or adversarially refined negatives. SugarCrepe is COCO-derived, so part of every fine-tuned model's gain is domain adaptation.
- **Deviation:** SugarCrepe replaced ARO as the secondary benchmark, decided before running either, because ARO has known text-only biases and SugarCrepe was built to remove them.
- Attribute-swap data mining was considered and skipped for time. `lora_hn_plaus` seed 2 was resumed from a checkpoint after a Colab quota limit (its `train_time_min` covers only the resumed epochs).

## Reproduce

Open `colab_runner.ipynb` (cells 1-6 each session; set `BRANCH`, name and email in cell 2). Pipeline, in order:
`scripts/prepare_coco_subset.py` -> `src/negatives.py` (train and val) -> `scripts/filter_plausible.py` -> `scripts/text_only_bias.py` -> `scripts/audit_negatives.py` ->
`scripts/run_all.py --tier main` and `--tier ablation` -> `scripts/probe_hn.py` -> `scripts/eval_sugarcrepe.py` -> `scripts/aggregate.py --tags` -> `scripts/visualize.py`.
Per-run results are in `results/*.json`, training logs in `logs/*.jsonl`, each result file records the git commit and full config. Checkpoints (about 4 MB each) stay on Drive.
