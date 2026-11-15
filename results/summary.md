## Winoground and retrieval (mean ± std over seeds, % ; zero-shot is a single run)

| config | seeds | W text | W image | W group | i2t R@1 | R@5 | R@10 | t2i R@1 | R@5 | R@10 | val L_CLIP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zeroshot | 1 | 31.0 | 11.2 | 9.0 | 68.9 | 90.0 | 95.2 | 49.0 | 77.8 | 87.9 | - |
| lora_baseline | 3 | 33.9 ± 1.4 | 12.3 ± 1.8 | 8.7 ± 0.8 | 75.0 ± 0.9 | 94.0 ± 0.4 | 97.4 ± 0.1 | 60.0 ± 0.2 | 86.6 ± 0.1 | 93.8 ± 0.1 | 0.372 ± 0.002 |
| lora_hn | 3 | 29.4 ± 0.1 | 11.9 ± 0.5 | 8.0 ± 0.4 | 73.1 ± 0.2 | 92.4 ± 0.3 | 96.5 ± 0.2 | 59.7 ± 0.2 | 86.1 ± 0.2 | 93.5 ± 0.0 | 0.402 ± 0.003 |
| lora_hn_attr | 1 | 33.2 | 11.2 | 7.5 | 73.2 | 92.9 | 96.7 | 59.3 | 86.1 | 93.5 | 0.398 |
| lora_hn_lam05 | 1 | 30.8 | 12.8 | 8.0 | 73.9 | 93.3 | 96.7 | 60.1 | 86.5 | 93.5 | 0.386 |
| lora_hn_lam2 | 1 | 28.2 | 11.8 | 8.2 | 71.5 | 90.6 | 96.2 | 59.5 | 85.6 | 93.4 | 0.431 |
| lora_hn_obj | 1 | 34.0 | 11.2 | 8.0 | 73.3 | 92.7 | 96.6 | 59.5 | 86.4 | 93.5 | 0.387 |
| lora_hn_plaus | 3 | 33.3 ± 1.0 | 12.0 ± 0.2 | 8.8 ± 0.4 | 72.8 ± 0.3 | 92.1 ± 0.5 | 96.4 ± 0.4 | 59.5 ± 0.4 | 86.3 ± 0.2 | 93.5 ± 0.2 | 0.400 ± 0.001 |
| lora_hn_rel | 3 | 29.2 ± 0.3 | 12.3 ± 0.4 | 7.8 ± 0.2 | 71.9 ± 0.3 | 92.2 ± 0.3 | 96.9 ± 0.3 | 59.7 ± 0.1 | 85.9 ± 0.1 | 93.4 ± 0.2 | 0.402 ± 0.004 |

val L_CLIP is comparable across conditions. Validation hn_acc is NOT in this table: it is measured on
different negative files for different conditions and can be inflated by text-only shortcuts.

## Paired bootstrap on Winoground (difference = test − ref, pp). Holm-adjusted p over the 9 pre-planned comparisons; other rows are exploratory (no correction)

| ref | test | metric | seeds | diff | 95% CI | seed std of diff | verdict | Holm p |
|---|---|---|---|---|---|---|---|---|
| lora_baseline | lora_hn | text | 3 | -4.50 | [-7.75, -1.33] | 1.32 | test lower | 0.040 |
| lora_baseline | lora_hn | image | 3 | -0.42 | [-2.33, +1.50] | 2.25 | no detectable difference | 1.000 |
| lora_baseline | lora_hn | group | 3 | -0.67 | [-2.50, +1.08] | 1.04 | no detectable difference | 1.000 |
| lora_baseline | lora_hn_plaus | text | 3 | -0.58 | [-2.92, +1.67] | 2.25 | no detectable difference | 1.000 |
| lora_baseline | lora_hn_plaus | image | 3 | -0.33 | [-2.33, +1.58] | 1.94 | no detectable difference | 1.000 |
| lora_baseline | lora_hn_plaus | group | 3 | +0.17 | [-1.50, +1.75] | 0.76 | no detectable difference | 1.000 |
| lora_hn_plaus | lora_hn | text | 3 | -3.92 | [-6.67, -1.25] | 1.15 | test lower | 0.040 |
| lora_hn_plaus | lora_hn | image | 3 | -0.08 | [-1.42, +1.33] | 0.52 | no detectable difference | 1.000 |
| lora_hn_plaus | lora_hn | group | 3 | -0.83 | [-2.08, +0.42] | 0.29 | no detectable difference | 1.000 |

## Retrieval differences (descriptive only: no per-example data, no test)

| ref | test | metric | mean diff (pp) | per-seed diff std (pp) |
|---|---|---|---|---|
| lora_baseline | lora_hn | i2t_r1 | -1.83 | 1.01 |
| lora_baseline | lora_hn | t2i_r1 | -0.28 | 0.16 |
| lora_baseline | lora_hn_plaus | i2t_r1 | -2.13 | 1.15 |
| lora_baseline | lora_hn_plaus | t2i_r1 | -0.47 | 0.27 |
| lora_hn_plaus | lora_hn | i2t_r1 | +0.30 | 0.17 |
| lora_hn_plaus | lora_hn | t2i_r1 | +0.19 | 0.37 |

## SugarCrepe accuracy (%, mean ± std over seeds; zero-shot is a single run)

| config | seeds | replace_obj | replace_att | replace_rel | swap_obj | swap_att | add_obj | add_att | mean |
|---|---|---|---|---|---|---|---|---|---|
| zeroshot | 1 | 90.7 | 80.3 | 69.4 | 61.2 | 64.0 | 77.1 | 69.5 | 73.2 |
| lora_baseline | 3 | 94.2 ± 0.2 | 85.8 ± 1.2 | 69.3 ± 0.2 | 63.1 ± 0.6 | 69.2 ± 0.8 | 86.5 ± 0.4 | 78.0 ± 0.6 | 78.0 ± 0.2 |
| lora_hn | 3 | 93.4 ± 0.3 | 86.3 ± 0.8 | 75.5 ± 0.3 | 68.2 ± 1.1 | 72.1 ± 0.9 | 85.6 ± 0.3 | 81.7 ± 1.0 | 80.4 ± 0.2 |
| lora_hn_plaus | 3 | 93.6 ± 0.2 | 86.8 ± 0.1 | 73.6 ± 0.5 | 66.5 ± 1.1 | 71.2 ± 0.5 | 86.8 ± 0.2 | 82.1 ± 1.1 | 80.1 ± 0.1 |
| lora_hn_rel | 3 | 92.9 ± 0.1 | 85.1 ± 0.3 | 75.6 ± 0.6 | 68.6 ± 0.4 | 71.1 ± 0.6 | 83.5 ± 0.1 | 80.2 ± 1.1 | 79.6 ± 0.1 |
| lora_hn_attr | 1 | 93.5 | 86.7 | 70.3 | 66.5 | 71.6 | 85.4 | 82.1 | 79.4 |
| lora_hn_obj | 1 | 93.7 | 84.0 | 69.7 | 63.7 | 67.4 | 88.5 | 79.0 | 78.0 |

## SugarCrepe paired bootstrap (diff = test − ref, pp). Holm-adjusted p over the 21 pre-planned comparisons; other rows exploratory

| ref | test | task | seeds | diff | 95% CI | seed std | verdict | Holm p |
|---|---|---|---|---|---|---|---|---|
| lora_baseline | lora_hn | replace_obj | 3 | -0.81 | [-1.55, -0.10] | 0.13 | test lower | 0.378 |
| lora_baseline | lora_hn | replace_att | 3 | +0.51 | [-1.10, +2.20] | 1.37 | no detectable difference | 1.000 |
| lora_baseline | lora_hn | replace_rel | 3 | +6.24 | [+4.55, +7.94] | 0.08 | test higher | 0.002 |
| lora_baseline | lora_hn | swap_obj | 3 | +5.03 | [-0.41, +10.34] | 1.55 | no detectable difference | 0.787 |
| lora_baseline | lora_hn | swap_att | 3 | +2.90 | [+0.40, +5.36] | 0.83 | test higher | 0.306 |
| lora_baseline | lora_hn | add_obj | 3 | -0.84 | [-1.75, +0.05] | 0.23 | no detectable difference | 0.787 |
| lora_baseline | lora_hn | add_att | 3 | +3.66 | [+1.49, +5.78] | 1.20 | test higher | 0.014 |
| lora_baseline | lora_hn_plaus | replace_obj | 3 | -0.65 | [-1.27, -0.04] | 0.13 | test lower | 0.424 |
| lora_baseline | lora_hn_plaus | replace_att | 3 | +0.93 | [-0.72, +2.58] | 1.18 | no detectable difference | 1.000 |
| lora_baseline | lora_hn_plaus | replace_rel | 3 | +4.27 | [+2.92, +5.64] | 0.33 | test higher | 0.002 |
| lora_baseline | lora_hn_plaus | swap_obj | 3 | +3.40 | [-1.50, +8.30] | 1.43 | no detectable difference | 1.000 |
| lora_baseline | lora_hn_plaus | swap_att | 3 | +2.00 | [-0.20, +4.21] | 0.98 | no detectable difference | 0.787 |
| lora_baseline | lora_hn_plaus | add_obj | 3 | +0.36 | [-0.40, +1.12] | 0.57 | no detectable difference | 1.000 |
| lora_baseline | lora_hn_plaus | add_att | 3 | +4.09 | [+2.12, +6.07] | 0.51 | test higher | 0.002 |
| lora_hn_plaus | lora_hn | replace_obj | 3 | -0.16 | [-0.67, +0.34] | 0.24 | no detectable difference | 1.000 |
| lora_hn_plaus | lora_hn | replace_att | 3 | -0.42 | [-1.69, +0.85] | 0.84 | no detectable difference | 1.000 |
| lora_hn_plaus | lora_hn | replace_rel | 3 | +1.97 | [+0.76, +3.20] | 0.27 | test higher | 0.022 |
| lora_hn_plaus | lora_hn | swap_obj | 3 | +1.63 | [-1.77, +5.03] | 1.87 | no detectable difference | 1.000 |
| lora_hn_plaus | lora_hn | swap_att | 3 | +0.90 | [-0.85, +2.65] | 0.60 | no detectable difference | 1.000 |
| lora_hn_plaus | lora_hn | add_obj | 3 | -1.20 | [-1.89, -0.50] | 0.39 | test lower | 0.020 |
| lora_hn_plaus | lora_hn | add_att | 3 | -0.43 | [-1.73, +0.87] | 1.38 | no detectable difference | 1.000 |
