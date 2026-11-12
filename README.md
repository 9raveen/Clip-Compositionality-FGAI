# Compositional Failures in CLIP: Improving Attribute-Object Binding via Hard-Negative Fine-Tuning

Foundations of Generative AI lab project. Team: <names>. 7-day schedule, Google Colab T4.

**Research question.** Does fine-tuning CLIP ViT-B/32 with structured, validated hard negatives improve compositional
understanding (Winoground, ARO) relative to (a) the pretrained model and (b) standard fine-tuning on the same data?
No improvement is assumed; every claim in the report must come from the result files.

## Experimental design
| Run | Tuning | Objective | Purpose |
|---|---|---|---|
| A `zeroshot` | none | none | pretrained reference |
| B `lora_baseline` | LoRA (r=16, q/v, both towers) | L_CLIP (lambda = 0) | effect of fine-tuning alone |
| C `lora_hn` | same LoRA | L_CLIP + lambda * L_HN | effect of hard negatives (B vs C) |
Same images, same captions per epoch, same seed, same evaluation protocol for all three. B and C use the same adapters
on purpose: comparing a full-fine-tuning baseline to a LoRA hard-negative model would confound two changes.
Main comparison: B and C with seeds 0, 1, 2. Ablations (1 seed): attribute-only, object-only, relation-only negatives, lambda in {0.5, 2}.
Optional: full fine-tuning, text-tower-only LoRA.

**Loss.** L_CLIP is symmetric InfoNCE over in-batch positives. L_HN ranks a caption above its validated negative for the same image:
-log sigmoid(s_pos - s_neg) at CLIP's temperature (`hn_loss: margin` is the alternative). Lambda is chosen on the validation split, never on a benchmark.

## Data and leakage prevention
- Images: COCO **train2017** subset, split by **image ID** into train / val / test (`data/splits.json`, disjoint, asserted in code). All captions of an image stay in one split.
- COCO **val2017 is never used** (SugarCrepe is built on it). Winoground and ARO are never used to generate negatives, tune lambda, or select checkpoints.
- Winoground is logged during training for monitoring only; reported numbers are final-epoch.
- ARO COCO-Order is skipped (COCO overlap). SugarCrepe (COCO-derived) is optional.

## Hard negatives
Five candidate types: `attr_replace`, `obj_replace`, `rel_replace` (single change), `obj_swap` (subject/object), `attr_swap` (attribute-object binding).
Categories for ablation: attribute = {attr_replace, attr_swap}; object = {obj_replace}; relation = {rel_replace, obj_swap}.
Candidates are filtered (identical, not a permutation, symmetric relation, same-noun swap, synonym replacement, ...).
Validation precision is measured by manual audit: `scripts/audit_negatives.py` -> report per-type precision here: **TODO**.

## Evaluation (same protocol for A, B, C)
Winoground text/image/group (primary, with bootstrap CI) | ARO VG-Relation and VG-Attribution (secondary, TODO loader) | COCO test retrieval R@1/5/10 (what fine-tuning cost) | SugarCrepe (optional).

## Reproduce
`colab_runner.ipynb` (cells in order). Configs: `configs/` (ablations inherit `_base.yaml`). Results: `results/*.json` include the git hash and config.

## Results
TODO: tables (mean +/- std over 3 seeds for B and C; single values for A and ablations), loss curves, Winoground comparison, qualitative examples.
