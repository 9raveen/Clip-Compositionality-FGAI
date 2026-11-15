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
| lora_baseline | lora_hn_attr | text | 1 | -0.25 | [-3.75, +3.25] | 0.00 | no detectable difference | exploratory |
| lora_baseline | lora_hn_attr | image | 1 | -3.00 | [-5.75, -0.50] | 0.00 | no detectable difference | exploratory |
| lora_baseline | lora_hn_attr | group | 1 | -1.50 | [-3.50, +0.50] | 0.00 | no detectable difference | exploratory |
| lora_baseline | lora_hn_obj | text | 1 | +0.50 | [-2.50, +3.50] | 0.00 | no detectable difference | exploratory |
| lora_baseline | lora_hn_obj | image | 1 | -3.00 | [-5.25, -0.75] | 0.00 | no detectable difference | exploratory |
| lora_baseline | lora_hn_obj | group | 1 | -1.00 | [-3.00, +1.00] | 0.00 | no detectable difference | exploratory |
| lora_baseline | lora_hn_rel | text | 3 | -4.67 | [-8.25, -1.17] | 1.61 | test lower | exploratory |
| lora_baseline | lora_hn_rel | image | 3 | +0.00 | [-1.92, +1.92] | 1.98 | no detectable difference | exploratory |
| lora_baseline | lora_hn_rel | group | 3 | -0.92 | [-2.67, +0.83] | 0.58 | no detectable difference | exploratory |
| lora_hn | lora_hn_attr | text | 1 | +3.75 | [-0.25, +7.75] | 0.00 | no detectable difference | exploratory |
| lora_hn | lora_hn_attr | image | 1 | -0.25 | [-2.75, +2.25] | 0.00 | no detectable difference | exploratory |
| lora_hn | lora_hn_attr | group | 1 | +0.00 | [-2.00, +2.00] | 0.00 | no detectable difference | exploratory |
| lora_hn | lora_hn_obj | text | 1 | +4.50 | [+0.75, +8.25] | 0.00 | test higher | exploratory |
| lora_hn | lora_hn_obj | image | 1 | -0.25 | [-2.50, +2.00] | 0.00 | no detectable difference | exploratory |
| lora_hn | lora_hn_obj | group | 1 | +0.50 | [-1.50, +2.50] | 0.00 | no detectable difference | exploratory |
| lora_hn | lora_hn_rel | text | 3 | -0.17 | [-2.25, +1.92] | 0.29 | no detectable difference | exploratory |
| lora_hn | lora_hn_rel | image | 3 | +0.42 | [-1.08, +1.92] | 0.63 | no detectable difference | exploratory |
| lora_hn | lora_hn_rel | group | 3 | -0.25 | [-1.67, +1.08] | 0.50 | no detectable difference | exploratory |

## Retrieval differences (descriptive only: no per-example data, no test)

| ref | test | metric | mean diff (pp) | per-seed diff std (pp) |
|---|---|---|---|---|
| lora_baseline | lora_hn | i2t_r1 | -1.83 | 1.01 |
| lora_baseline | lora_hn | t2i_r1 | -0.28 | 0.16 |
| lora_baseline | lora_hn_plaus | i2t_r1 | -2.13 | 1.15 |
| lora_baseline | lora_hn_plaus | t2i_r1 | -0.47 | 0.27 |
| lora_hn_plaus | lora_hn | i2t_r1 | +0.30 | 0.17 |
| lora_hn_plaus | lora_hn | t2i_r1 | +0.19 | 0.37 |
| lora_baseline | lora_hn_attr | i2t_r1 | -1.60 | nan |
| lora_baseline | lora_hn_attr | t2i_r1 | -0.92 | nan |
| lora_baseline | lora_hn_obj | i2t_r1 | -1.50 | nan |
| lora_baseline | lora_hn_obj | t2i_r1 | -0.72 | nan |
| lora_baseline | lora_hn_rel | i2t_r1 | -3.03 | 0.60 |
| lora_baseline | lora_hn_rel | t2i_r1 | -0.33 | 0.25 |
| lora_hn | lora_hn_attr | i2t_r1 | +0.10 | nan |
| lora_hn | lora_hn_attr | t2i_r1 | -0.64 | nan |
| lora_hn | lora_hn_obj | i2t_r1 | +0.20 | nan |
| lora_hn | lora_hn_obj | t2i_r1 | -0.44 | nan |
| lora_hn | lora_hn_rel | i2t_r1 | -1.20 | 0.44 |
| lora_hn | lora_hn_rel | t2i_r1 | -0.05 | 0.25 |

## Winoground by tag (mean ± std over seeds, %)

| config | Both text | Both image | Both group | Object text | Object image | Object group | Relation text | Relation image | Relation group |
|---|---|---|---|---|---|---|---|---|---|
| lora_baseline | 67.9 ± 4.4 | 35.9 ± 2.2 | 30.8 ± 0.0 | 43.7 ± 1.1 | 14.4 ± 2.3 | 10.4 ± 1.8 | 24.2 ± 1.4 | 8.4 ± 1.5 | 5.2 ± 0.7 |
| lora_hn | 66.7 ± 2.2 | 33.3 ± 2.2 | 29.5 ± 2.2 | 34.8 ± 0.0 | 12.3 ± 0.4 | 8.3 ± 0.8 | 22.0 ± 0.2 | 9.3 ± 0.5 | 5.4 ± 0.5 |
| lora_hn_attr | 65.4 | 30.8 | 26.9 | 41.1 | 13.5 | 8.5 | 24.9 | 7.7 | 4.7 |
| lora_hn_lam05 | 61.5 | 30.8 | 26.9 | 37.6 | 14.9 | 9.2 | 23.2 | 9.4 | 5.2 |
| lora_hn_lam2 | 69.2 | 34.6 | 30.8 | 31.9 | 10.6 | 7.1 | 21.5 | 9.9 | 6.4 |
| lora_hn_obj | 73.1 | 26.9 | 26.9 | 40.4 | 14.2 | 10.6 | 25.8 | 7.7 | 4.3 |
| lora_hn_plaus | 73.1 ± 0.0 | 28.2 ± 2.2 | 28.2 ± 2.2 | 42.1 ± 2.5 | 13.7 ± 0.8 | 9.9 ± 0.0 | 23.6 ± 0.7 | 9.2 ± 0.7 | 6.0 ± 0.4 |
| lora_hn_rel | 65.4 ± 6.7 | 35.9 ± 2.2 | 30.8 ± 3.8 | 35.5 ± 1.2 | 13.0 ± 1.8 | 8.3 ± 0.8 | 21.5 ± 1.5 | 9.3 ± 1.0 | 4.9 ± 0.9 |
| n examples | 26 | 26 | 26 | 141 | 141 | 141 | 233 | 233 | 233 |

## Paired bootstrap by tag (EXPLORATORY, uncorrected for multiple comparisons; diff = test − ref, pp). Tags with n < 30 must not be interpreted.

| ref | test | tag (n) | metric | seeds | diff | 95% CI | seed std of diff | verdict |
|---|---|---|---|---|---|---|---|---|
| lora_baseline | lora_hn | Both (26) | text | 3 | -1.28 | [-11.54, +7.69] | 2.22 | no detectable difference |
| lora_baseline | lora_hn | Both (26) | image | 3 | -2.56 | [-11.54, +3.85] | 4.44 | no detectable difference |
| lora_baseline | lora_hn | Both (26) | group | 3 | -1.28 | [-11.54, +7.69] | 2.22 | no detectable difference |
| lora_baseline | lora_hn | Object (141) | text | 3 | -8.98 | [-15.37, -2.84] | 1.08 | test lower |
| lora_baseline | lora_hn | Object (141) | image | 3 | -2.13 | [-5.44, +0.95] | 2.56 | no detectable difference |
| lora_baseline | lora_hn | Object (141) | group | 3 | -2.13 | [-5.91, +1.18] | 2.56 | no detectable difference |
| lora_baseline | lora_hn | Relation (233) | text | 3 | -2.15 | [-5.87, +1.43] | 1.49 | no detectable difference |
| lora_baseline | lora_hn | Relation (233) | image | 3 | +0.86 | [-1.57, +3.43] | 1.97 | no detectable difference |
| lora_baseline | lora_hn | Relation (233) | group | 3 | +0.29 | [-1.72, +2.15] | 0.25 | no detectable difference |
| lora_baseline | lora_hn_plaus | Both (26) | text | 3 | +5.13 | [+0.00, +12.82] | 4.44 | no detectable difference |
| lora_baseline | lora_hn_plaus | Both (26) | image | 3 | -7.69 | [-19.23, +0.00] | 3.85 | no detectable difference |
| lora_baseline | lora_hn_plaus | Both (26) | group | 3 | -2.56 | [-11.54, +3.85] | 2.22 | no detectable difference |
| lora_baseline | lora_hn_plaus | Object (141) | text | 3 | -1.65 | [-5.91, +2.60] | 3.57 | no detectable difference |
| lora_baseline | lora_hn_plaus | Object (141) | image | 3 | -0.71 | [-4.49, +2.84] | 1.88 | no detectable difference |
| lora_baseline | lora_hn_plaus | Object (141) | group | 3 | -0.47 | [-3.55, +2.36] | 1.78 | no detectable difference |
| lora_baseline | lora_hn_plaus | Relation (233) | text | 3 | -0.57 | [-3.29, +2.15] | 1.31 | no detectable difference |
| lora_baseline | lora_hn_plaus | Relation (233) | image | 3 | +0.72 | [-1.57, +2.86] | 2.16 | no detectable difference |
| lora_baseline | lora_hn_plaus | Relation (233) | group | 3 | +0.86 | [-1.14, +2.86] | 0.43 | no detectable difference |
| lora_hn_plaus | lora_hn | Both (26) | text | 3 | -6.41 | [-15.38, +0.00] | 2.22 | no detectable difference |
| lora_hn_plaus | lora_hn | Both (26) | image | 3 | +5.13 | [+0.00, +14.10] | 2.22 | no detectable difference |
| lora_hn_plaus | lora_hn | Both (26) | group | 3 | +1.28 | [+0.00, +3.85] | 2.22 | no detectable difference |
| lora_hn_plaus | lora_hn | Object (141) | text | 3 | -7.33 | [-12.53, -2.60] | 2.49 | test lower |
| lora_hn_plaus | lora_hn | Object (141) | image | 3 | -1.42 | [-3.78, +0.95] | 1.23 | no detectable difference |
| lora_hn_plaus | lora_hn | Object (141) | group | 3 | -1.65 | [-4.26, +0.95] | 0.82 | no detectable difference |
| lora_hn_plaus | lora_hn | Relation (233) | text | 3 | -1.57 | [-4.87, +1.72] | 0.99 | no detectable difference |
| lora_hn_plaus | lora_hn | Relation (233) | image | 3 | +0.14 | [-1.43, +1.72] | 0.25 | no detectable difference |
| lora_hn_plaus | lora_hn | Relation (233) | group | 3 | -0.57 | [-2.00, +0.86] | 0.25 | no detectable difference |
| lora_baseline | lora_hn_attr | Both (26) | text | 1 | +0.00 | [-11.54, +11.54] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_attr | Both (26) | image | 1 | -7.69 | [-19.23, +0.00] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_attr | Both (26) | group | 1 | -3.85 | [-11.54, +0.00] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_attr | Object (141) | text | 1 | -2.84 | [-9.93, +4.26] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_attr | Object (141) | image | 1 | -3.55 | [-8.51, +0.71] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_attr | Object (141) | group | 1 | -3.55 | [-8.51, +0.71] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_attr | Relation (233) | text | 1 | +1.29 | [-2.58, +5.58] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_attr | Relation (233) | image | 1 | -2.15 | [-5.58, +0.86] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_attr | Relation (233) | group | 1 | +0.00 | [-2.15, +2.15] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_obj | Both (26) | text | 1 | +7.69 | [+0.00, +19.23] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_obj | Both (26) | image | 1 | -11.54 | [-23.08, +0.00] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_obj | Both (26) | group | 1 | -3.85 | [-11.54, +0.00] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_obj | Object (141) | text | 1 | -3.55 | [-8.51, +1.42] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_obj | Object (141) | image | 1 | -2.84 | [-7.09, +0.71] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_obj | Object (141) | group | 1 | -1.42 | [-5.67, +2.84] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_obj | Relation (233) | text | 1 | +2.15 | [-1.72, +6.01] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_obj | Relation (233) | image | 1 | -2.15 | [-5.15, +0.43] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_obj | Relation (233) | group | 1 | -0.43 | [-2.58, +1.72] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_rel | Both (26) | text | 3 | -2.56 | [-10.26, +5.13] | 2.22 | no detectable difference |
| lora_baseline | lora_hn_rel | Both (26) | image | 3 | +0.00 | [-3.85, +3.85] | 3.85 | no detectable difference |
| lora_baseline | lora_hn_rel | Both (26) | group | 3 | +0.00 | [-7.69, +7.69] | 3.85 | no detectable difference |
| lora_baseline | lora_hn_rel | Object (141) | text | 3 | -8.27 | [-15.37, -1.42] | 0.41 | test lower |
| lora_baseline | lora_hn_rel | Object (141) | image | 3 | -1.42 | [-5.20, +2.13] | 3.95 | no detectable difference |
| lora_baseline | lora_hn_rel | Object (141) | group | 3 | -2.13 | [-6.15, +1.42] | 2.56 | no detectable difference |
| lora_baseline | lora_hn_rel | Relation (233) | text | 3 | -2.72 | [-6.87, +1.29] | 2.92 | no detectable difference |
| lora_baseline | lora_hn_rel | Relation (233) | image | 3 | +0.86 | [-1.43, +3.29] | 0.74 | no detectable difference |
| lora_baseline | lora_hn_rel | Relation (233) | group | 3 | -0.29 | [-2.00, +1.29] | 0.99 | no detectable difference |
| lora_hn | lora_hn_attr | Both (26) | text | 1 | +0.00 | [-11.54, +11.54] | 0.00 | no detectable difference |
| lora_hn | lora_hn_attr | Both (26) | image | 1 | +0.00 | [+0.00, +0.00] | 0.00 | no detectable difference |
| lora_hn | lora_hn_attr | Both (26) | group | 1 | +0.00 | [+0.00, +0.00] | 0.00 | no detectable difference |
| lora_hn | lora_hn_attr | Object (141) | text | 1 | +6.38 | [-0.71, +13.48] | 0.00 | no detectable difference |
| lora_hn | lora_hn_attr | Object (141) | image | 1 | +1.42 | [-2.84, +5.67] | 0.00 | no detectable difference |
| lora_hn | lora_hn_attr | Object (141) | group | 1 | +0.71 | [-3.55, +4.96] | 0.00 | no detectable difference |
| lora_hn | lora_hn_attr | Relation (233) | text | 1 | +2.58 | [-2.58, +7.73] | 0.00 | no detectable difference |
| lora_hn | lora_hn_attr | Relation (233) | image | 1 | -1.29 | [-4.72, +1.72] | 0.00 | no detectable difference |
| lora_hn | lora_hn_attr | Relation (233) | group | 1 | -0.43 | [-2.58, +1.72] | 0.00 | no detectable difference |
| lora_hn | lora_hn_obj | Both (26) | text | 1 | +7.69 | [+0.00, +19.23] | 0.00 | no detectable difference |
| lora_hn | lora_hn_obj | Both (26) | image | 1 | -3.85 | [-11.54, +0.00] | 0.00 | no detectable difference |
| lora_hn | lora_hn_obj | Both (26) | group | 1 | +0.00 | [+0.00, +0.00] | 0.00 | no detectable difference |
| lora_hn | lora_hn_obj | Object (141) | text | 1 | +5.67 | [-0.71, +12.06] | 0.00 | no detectable difference |
| lora_hn | lora_hn_obj | Object (141) | image | 1 | +2.13 | [-2.13, +6.38] | 0.00 | no detectable difference |
| lora_hn | lora_hn_obj | Object (141) | group | 1 | +2.84 | [-0.71, +7.09] | 0.00 | no detectable difference |
| lora_hn | lora_hn_obj | Relation (233) | text | 1 | +3.43 | [-1.29, +8.17] | 0.00 | no detectable difference |
| lora_hn | lora_hn_obj | Relation (233) | image | 1 | -1.29 | [-3.86, +1.29] | 0.00 | no detectable difference |
| lora_hn | lora_hn_obj | Relation (233) | group | 1 | -0.86 | [-3.43, +1.29] | 0.00 | no detectable difference |
| lora_hn | lora_hn_rel | Both (26) | text | 3 | -1.28 | [-5.13, +2.56] | 4.44 | no detectable difference |
| lora_hn | lora_hn_rel | Both (26) | image | 3 | +2.56 | [+0.00, +7.69] | 2.22 | no detectable difference |
| lora_hn | lora_hn_rel | Both (26) | group | 3 | +1.28 | [+0.00, +3.85] | 2.22 | no detectable difference |
| lora_hn | lora_hn_rel | Object (141) | text | 3 | +0.71 | [-2.84, +4.49] | 1.23 | no detectable difference |
| lora_hn | lora_hn_rel | Object (141) | image | 3 | +0.71 | [-1.89, +3.31] | 1.42 | no detectable difference |
| lora_hn | lora_hn_rel | Object (141) | group | 3 | -0.00 | [-2.84, +2.84] | 0.00 | no detectable difference |
| lora_hn | lora_hn_rel | Relation (233) | text | 3 | -0.57 | [-3.58, +2.29] | 1.51 | no detectable difference |
| lora_hn | lora_hn_rel | Relation (233) | image | 3 | +0.00 | [-2.00, +2.00] | 1.49 | no detectable difference |
| lora_hn | lora_hn_rel | Relation (233) | group | 3 | -0.57 | [-2.29, +0.86] | 0.89 | no detectable difference |

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
| lora_baseline | lora_hn_attr | replace_obj | 1 | -0.85 | [-1.63, -0.06] | 0.00 | test lower | exploratory |
| lora_baseline | lora_hn_attr | replace_att | 1 | +2.16 | [+0.00, +4.31] | 0.00 | no detectable difference | exploratory |
| lora_baseline | lora_hn_attr | replace_rel | 1 | +1.07 | [-0.50, +2.63] | 0.00 | no detectable difference | exploratory |
| lora_baseline | lora_hn_attr | swap_obj | 1 | +4.08 | [-0.82, +8.98] | 0.00 | no detectable difference | exploratory |
| lora_baseline | lora_hn_attr | swap_att | 1 | +1.95 | [-0.75, +4.80] | 0.00 | no detectable difference | exploratory |
| lora_baseline | lora_hn_attr | add_obj | 1 | -1.55 | [-2.67, -0.48] | 0.00 | test lower | exploratory |
| lora_baseline | lora_hn_attr | add_att | 1 | +4.77 | [+2.31, +7.37] | 0.00 | test higher | exploratory |
| lora_baseline | lora_hn_obj | replace_obj | 1 | -0.61 | [-1.39, +0.18] | 0.00 | no detectable difference | exploratory |
| lora_baseline | lora_hn_obj | replace_att | 1 | -0.51 | [-2.03, +1.02] | 0.00 | no detectable difference | exploratory |
| lora_baseline | lora_hn_obj | replace_rel | 1 | +0.43 | [-0.85, +1.78] | 0.00 | no detectable difference | exploratory |
| lora_baseline | lora_hn_obj | swap_obj | 1 | +1.22 | [-3.67, +6.12] | 0.00 | no detectable difference | exploratory |
| lora_baseline | lora_hn_obj | swap_att | 1 | -2.25 | [-4.65, +0.15] | 0.00 | no detectable difference | exploratory |
| lora_baseline | lora_hn_obj | add_obj | 1 | +1.50 | [+0.63, +2.38] | 0.00 | test higher | exploratory |
| lora_baseline | lora_hn_obj | add_att | 1 | +1.73 | [+0.00, +3.61] | 0.00 | no detectable difference | exploratory |
| lora_baseline | lora_hn_rel | replace_obj | 3 | -1.33 | [-2.04, -0.65] | 0.18 | test lower | exploratory |
| lora_baseline | lora_hn_rel | replace_att | 3 | -0.76 | [-2.33, +0.80] | 0.92 | no detectable difference | exploratory |
| lora_baseline | lora_hn_rel | replace_rel | 3 | +6.31 | [+4.53, +8.13] | 0.68 | test higher | exploratory |
| lora_baseline | lora_hn_rel | swap_obj | 3 | +5.44 | [+0.14, +10.75] | 0.24 | test higher | exploratory |
| lora_baseline | lora_hn_rel | swap_att | 3 | +1.90 | [-0.65, +4.45] | 1.35 | no detectable difference | exploratory |
| lora_baseline | lora_hn_rel | add_obj | 3 | -2.99 | [-3.93, -2.09] | 0.52 | test lower | exploratory |
| lora_baseline | lora_hn_rel | add_att | 3 | +2.12 | [-0.10, +4.29] | 1.69 | no detectable difference | exploratory |
| lora_hn | lora_hn_attr | replace_obj | 1 | +0.06 | [-0.73, +0.85] | 0.00 | no detectable difference | exploratory |
| lora_hn | lora_hn_attr | replace_att | 1 | +0.51 | [-1.27, +2.29] | 0.00 | no detectable difference | exploratory |
| lora_hn | lora_hn_attr | replace_rel | 1 | -5.26 | [-7.18, -3.34] | 0.00 | test lower | exploratory |
| lora_hn | lora_hn_attr | swap_obj | 1 | -2.04 | [-7.35, +3.27] | 0.00 | no detectable difference | exploratory |
| lora_hn | lora_hn_attr | swap_att | 1 | +0.00 | [-2.55, +2.70] | 0.00 | no detectable difference | exploratory |
| lora_hn | lora_hn_attr | add_obj | 1 | -0.63 | [-1.65, +0.39] | 0.00 | no detectable difference | exploratory |
| lora_hn | lora_hn_attr | add_att | 1 | +0.14 | [-2.02, +2.17] | 0.00 | no detectable difference | exploratory |
| lora_hn | lora_hn_obj | replace_obj | 1 | +0.30 | [-0.48, +1.09] | 0.00 | no detectable difference | exploratory |
| lora_hn | lora_hn_obj | replace_att | 1 | -2.16 | [-4.19, -0.25] | 0.00 | test lower | exploratory |
| lora_hn | lora_hn_obj | replace_rel | 1 | -5.90 | [-7.82, -3.98] | 0.00 | test lower | exploratory |
| lora_hn | lora_hn_obj | swap_obj | 1 | -4.90 | [-10.61, +0.82] | 0.00 | no detectable difference | exploratory |
| lora_hn | lora_hn_obj | swap_att | 1 | -4.20 | [-6.91, -1.50] | 0.00 | test lower | exploratory |
| lora_hn | lora_hn_obj | add_obj | 1 | +2.42 | [+1.41, +3.44] | 0.00 | test higher | exploratory |
| lora_hn | lora_hn_obj | add_att | 1 | -2.89 | [-5.06, -0.72] | 0.00 | test lower | exploratory |
| lora_hn | lora_hn_rel | replace_obj | 3 | -0.52 | [-1.09, +0.02] | 0.29 | no detectable difference | exploratory |
| lora_hn | lora_hn_rel | replace_att | 3 | -1.27 | [-2.75, +0.21] | 0.71 | no detectable difference | exploratory |
| lora_hn | lora_hn_rel | replace_rel | 3 | +0.07 | [-0.92, +1.07] | 0.61 | no detectable difference | exploratory |
| lora_hn | lora_hn_rel | swap_obj | 3 | +0.41 | [-2.18, +2.99] | 1.41 | no detectable difference | exploratory |
| lora_hn | lora_hn_rel | swap_att | 3 | -1.00 | [-2.90, +0.90] | 1.20 | no detectable difference | exploratory |
| lora_hn | lora_hn_rel | add_obj | 3 | -2.15 | [-2.88, -1.44] | 0.47 | test lower | exploratory |
| lora_hn | lora_hn_rel | add_att | 3 | -1.54 | [-3.18, +0.10] | 1.01 | no detectable difference | exploratory |
