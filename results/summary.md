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
| lora_hn_objswap | 2 | 28.1 ± 0.5 | 12.4 ± 0.2 | 8.1 ± 0.5 | 73.9 ± 0.4 | 92.5 ± 0.6 | 96.9 ± 0.1 | 59.8 ± 0.1 | 86.2 ± 0.1 | 93.3 ± 0.1 | 0.405 ± 0.001 |
| lora_hn_plaus | 3 | 33.3 ± 1.0 | 12.0 ± 0.2 | 8.8 ± 0.4 | 72.8 ± 0.3 | 92.1 ± 0.5 | 96.4 ± 0.4 | 59.5 ± 0.4 | 86.3 ± 0.2 | 93.5 ± 0.2 | 0.400 ± 0.001 |
| lora_hn_rel | 3 | 29.2 ± 0.3 | 12.3 ± 0.4 | 7.8 ± 0.2 | 71.9 ± 0.3 | 92.2 ± 0.3 | 96.9 ± 0.3 | 59.7 ± 0.1 | 85.9 ± 0.1 | 93.4 ± 0.2 | 0.402 ± 0.004 |
| lora_hn_relrep | 2 | 32.1 ± 0.2 | 12.0 ± 1.4 | 8.8 ± 1.4 | 73.6 ± 1.3 | 92.5 ± 0.1 | 96.7 ± 0.3 | 59.5 ± 0.1 | 86.1 ± 0.3 | 93.4 ± 0.0 | 0.381 ± 0.003 |

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
| lora_baseline | lora_hn_objswap | text | 2 | -6.38 | [-9.88, -2.88] | 0.88 | test lower | exploratory |
| lora_baseline | lora_hn_objswap | image | 2 | -0.75 | [-2.62, +1.00] | 1.41 | no detectable difference | exploratory |
| lora_baseline | lora_hn_objswap | group | 2 | -1.00 | [-2.75, +0.62] | 0.71 | no detectable difference | exploratory |
| lora_baseline | lora_hn_relrep | text | 2 | -2.38 | [-5.12, +0.25] | 1.59 | no detectable difference | exploratory |
| lora_baseline | lora_hn_relrep | image | 2 | -1.12 | [-3.00, +0.62] | 0.18 | no detectable difference | exploratory |
| lora_baseline | lora_hn_relrep | group | 2 | -0.38 | [-2.00, +1.25] | 1.59 | no detectable difference | exploratory |
| lora_hn | lora_hn_objswap | text | 2 | -1.38 | [-4.00, +1.25] | 0.53 | no detectable difference | exploratory |
| lora_hn | lora_hn_objswap | image | 2 | +0.75 | [-1.12, +2.62] | 0.35 | no detectable difference | exploratory |
| lora_hn | lora_hn_objswap | group | 2 | +0.25 | [-1.50, +1.88] | 1.06 | no detectable difference | exploratory |
| lora_hn | lora_hn_relrep | text | 2 | +2.62 | [-0.38, +5.62] | 0.18 | no detectable difference | exploratory |
| lora_hn | lora_hn_relrep | image | 2 | +0.38 | [-1.50, +2.25] | 1.59 | no detectable difference | exploratory |
| lora_hn | lora_hn_relrep | group | 2 | +0.88 | [-0.75, +2.50] | 1.94 | no detectable difference | exploratory |

## Retrieval differences (descriptive only: no per-example data, no test)

| ref | test | metric | mean diff (pp) | per-seed diff std (pp) |
|---|---|---|---|---|
| lora_baseline | lora_hn | i2t_r1 | -1.83 | 1.01 |
| lora_baseline | lora_hn | t2i_r1 | -0.28 | 0.16 |
| lora_baseline | lora_hn_plaus | i2t_r1 | -2.13 | 1.15 |
| lora_baseline | lora_hn_plaus | t2i_r1 | -0.47 | 0.27 |
| lora_hn_plaus | lora_hn | i2t_r1 | +0.30 | 0.17 |
| lora_hn_plaus | lora_hn | t2i_r1 | +0.19 | 0.37 |
| lora_baseline | lora_hn_objswap | i2t_r1 | -0.60 | 0.00 |
| lora_baseline | lora_hn_objswap | t2i_r1 | -0.19 | 0.10 |
| lora_baseline | lora_hn_relrep | i2t_r1 | -0.90 | 0.85 |
| lora_baseline | lora_hn_relrep | t2i_r1 | -0.50 | 0.34 |
| lora_hn | lora_hn_objswap | i2t_r1 | +0.70 | 0.57 |
| lora_hn | lora_hn_objswap | t2i_r1 | +0.17 | 0.21 |
| lora_hn | lora_hn_relrep | i2t_r1 | +0.40 | 1.41 |
| lora_hn | lora_hn_relrep | t2i_r1 | -0.14 | 0.45 |

## Winoground by tag (mean ± std over seeds, %)

| config | Both text | Both image | Both group | Object text | Object image | Object group | Relation text | Relation image | Relation group |
|---|---|---|---|---|---|---|---|---|---|
| lora_baseline | 67.9 ± 4.4 | 35.9 ± 2.2 | 30.8 ± 0.0 | 43.7 ± 1.1 | 14.4 ± 2.3 | 10.4 ± 1.8 | 24.2 ± 1.4 | 8.4 ± 1.5 | 5.2 ± 0.7 |
| lora_hn | 66.7 ± 2.2 | 33.3 ± 2.2 | 29.5 ± 2.2 | 34.8 ± 0.0 | 12.3 ± 0.4 | 8.3 ± 0.8 | 22.0 ± 0.2 | 9.3 ± 0.5 | 5.4 ± 0.5 |
| lora_hn_attr | 65.4 | 30.8 | 26.9 | 41.1 | 13.5 | 8.5 | 24.9 | 7.7 | 4.7 |
| lora_hn_lam05 | 61.5 | 30.8 | 26.9 | 37.6 | 14.9 | 9.2 | 23.2 | 9.4 | 5.2 |
| lora_hn_lam2 | 69.2 | 34.6 | 30.8 | 31.9 | 10.6 | 7.1 | 21.5 | 9.9 | 6.4 |
| lora_hn_obj | 73.1 | 26.9 | 26.9 | 40.4 | 14.2 | 10.6 | 25.8 | 7.7 | 4.3 |
| lora_hn_objswap | 65.4 ± 0.0 | 28.8 ± 2.7 | 25.0 ± 2.7 | 31.6 ± 0.5 | 14.5 ± 0.5 | 8.2 ± 0.5 | 21.9 ± 0.6 | 9.2 ± 0.3 | 6.2 ± 0.3 |
| lora_hn_plaus | 73.1 ± 0.0 | 28.2 ± 2.2 | 28.2 ± 2.2 | 42.1 ± 2.5 | 13.7 ± 0.8 | 9.9 ± 0.0 | 23.6 ± 0.7 | 9.2 ± 0.7 | 6.0 ± 0.4 |
| lora_hn_rel | 65.4 ± 6.7 | 35.9 ± 2.2 | 30.8 ± 3.8 | 35.5 ± 1.2 | 13.0 ± 1.8 | 8.3 ± 0.8 | 21.5 ± 1.5 | 9.3 ± 1.0 | 4.9 ± 0.9 |
| lora_hn_relrep | 69.2 ± 0.0 | 36.5 ± 2.7 | 34.6 ± 0.0 | 40.8 ± 0.5 | 13.5 ± 3.0 | 9.9 ± 3.0 | 22.7 ± 0.0 | 8.4 ± 0.9 | 5.2 ± 0.6 |
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
| lora_baseline | lora_hn_objswap | Both (26) | text | 2 | -3.85 | [-17.31, +7.69] | 5.44 | no detectable difference |
| lora_baseline | lora_hn_objswap | Both (26) | image | 2 | -7.69 | [-17.31, +0.00] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_objswap | Both (26) | group | 2 | -5.77 | [-15.38, +0.00] | 2.72 | no detectable difference |
| lora_baseline | lora_hn_objswap | Object (141) | text | 2 | -12.77 | [-19.86, -6.03] | 0.00 | test lower |
| lora_baseline | lora_hn_objswap | Object (141) | image | 2 | -0.71 | [-3.55, +2.13] | 2.01 | no detectable difference |
| lora_baseline | lora_hn_objswap | Object (141) | group | 2 | -3.19 | [-7.45, +0.35] | 0.50 | no detectable difference |
| lora_baseline | lora_hn_objswap | Relation (233) | text | 2 | -2.79 | [-6.87, +1.07] | 0.91 | no detectable difference |
| lora_baseline | lora_hn_objswap | Relation (233) | image | 2 | +0.00 | [-2.36, +2.15] | 1.21 | no detectable difference |
| lora_baseline | lora_hn_objswap | Relation (233) | group | 2 | +0.86 | [-0.43, +2.15] | 1.21 | no detectable difference |
| lora_baseline | lora_hn_relrep | Both (26) | text | 2 | +0.00 | [-11.54, +9.62] | 5.44 | no detectable difference |
| lora_baseline | lora_hn_relrep | Both (26) | image | 2 | +0.00 | [-5.77, +5.77] | 5.44 | no detectable difference |
| lora_baseline | lora_hn_relrep | Both (26) | group | 2 | +3.85 | [+0.00, +11.54] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_relrep | Object (141) | text | 2 | -3.55 | [-8.51, +1.42] | 1.00 | no detectable difference |
| lora_baseline | lora_hn_relrep | Object (141) | image | 2 | -1.77 | [-5.33, +1.77] | 0.50 | no detectable difference |
| lora_baseline | lora_hn_relrep | Object (141) | group | 2 | -1.42 | [-5.32, +1.77] | 2.01 | no detectable difference |
| lora_baseline | lora_hn_relrep | Relation (233) | text | 2 | -1.93 | [-5.15, +1.29] | 1.52 | no detectable difference |
| lora_baseline | lora_hn_relrep | Relation (233) | image | 2 | -0.86 | [-3.00, +1.29] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_relrep | Relation (233) | group | 2 | -0.21 | [-1.93, +1.29] | 1.52 | no detectable difference |
| lora_hn | lora_hn_objswap | Both (26) | text | 2 | -1.92 | [-11.54, +5.77] | 2.72 | no detectable difference |
| lora_hn | lora_hn_objswap | Both (26) | image | 2 | -3.85 | [-9.62, +0.00] | 5.44 | no detectable difference |
| lora_hn | lora_hn_objswap | Both (26) | group | 2 | -3.85 | [-9.62, +0.00] | 5.44 | no detectable difference |
| lora_hn | lora_hn_objswap | Object (141) | text | 2 | -3.19 | [-8.16, +1.77] | 0.50 | no detectable difference |
| lora_hn | lora_hn_objswap | Object (141) | image | 2 | +2.48 | [-1.06, +6.03] | 0.50 | no detectable difference |
| lora_hn | lora_hn_objswap | Object (141) | group | 2 | +0.35 | [-3.55, +3.90] | 0.50 | no detectable difference |
| lora_hn | lora_hn_objswap | Relation (233) | text | 2 | -0.21 | [-3.43, +2.79] | 0.91 | no detectable difference |
| lora_hn | lora_hn_objswap | Relation (233) | image | 2 | +0.21 | [-1.93, +2.58] | 0.30 | no detectable difference |
| lora_hn | lora_hn_objswap | Relation (233) | group | 2 | +0.64 | [-1.29, +2.58] | 0.91 | no detectable difference |
| lora_hn | lora_hn_relrep | Both (26) | text | 2 | +1.92 | [-9.62, +15.38] | 2.72 | no detectable difference |
| lora_hn | lora_hn_relrep | Both (26) | image | 2 | +3.85 | [-3.85, +13.46] | 0.00 | no detectable difference |
| lora_hn | lora_hn_relrep | Both (26) | group | 2 | +5.77 | [+0.00, +15.38] | 2.72 | no detectable difference |
| lora_hn | lora_hn_relrep | Object (141) | text | 2 | +6.03 | [+0.35, +11.70] | 0.50 | test higher |
| lora_hn | lora_hn_relrep | Object (141) | image | 2 | +1.42 | [-1.77, +4.96] | 3.01 | no detectable difference |
| lora_hn | lora_hn_relrep | Object (141) | group | 2 | +2.13 | [-1.06, +5.32] | 3.01 | no detectable difference |
| lora_hn | lora_hn_relrep | Relation (233) | text | 2 | +0.64 | [-2.79, +4.08] | 0.30 | no detectable difference |
| lora_hn | lora_hn_relrep | Relation (233) | image | 2 | -0.64 | [-2.79, +1.50] | 0.91 | no detectable difference |
| lora_hn | lora_hn_relrep | Relation (233) | group | 2 | -0.43 | [-2.15, +1.29] | 1.21 | no detectable difference |

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
| lora_hn_objswap | 2 | 93.0 ± 0.1 | 85.2 ± 0.5 | 69.8 ± 1.2 | 69.6 ± 2.0 | 71.0 ± 0.0 | 84.0 ± 0.2 | 83.1 ± 1.0 | 79.4 ± 0.4 |
| lora_hn_relrep | 2 | 93.4 ± 0.1 | 83.8 ± 0.3 | 75.9 ± 0.1 | 63.9 ± 0.3 | 67.4 ± 1.1 | 84.5 ± 0.5 | 78.7 ± 0.9 | 78.2 ± 0.4 |

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
| lora_baseline | lora_hn_objswap | replace_obj | 2 | -1.30 | [-2.03, -0.61] | 0.04 | test lower | exploratory |
| lora_baseline | lora_hn_objswap | replace_att | 2 | -0.25 | [-1.90, +1.40] | 0.72 | no detectable difference | exploratory |
| lora_baseline | lora_hn_objswap | replace_rel | 2 | +0.68 | [-1.10, +2.42] | 1.36 | no detectable difference | exploratory |
| lora_baseline | lora_hn_objswap | swap_obj | 2 | +6.53 | [+0.61, +12.45] | 1.15 | test higher | exploratory |
| lora_baseline | lora_hn_objswap | swap_att | 2 | +1.35 | [-1.20, +3.90] | 0.00 | no detectable difference | exploratory |
| lora_baseline | lora_hn_objswap | add_obj | 2 | -2.64 | [-3.61, -1.72] | 0.24 | test lower | exploratory |
| lora_baseline | lora_hn_objswap | add_att | 2 | +5.20 | [+3.03, +7.37] | 1.84 | test higher | exploratory |
| lora_baseline | lora_hn_relrep | replace_obj | 2 | -0.97 | [-1.57, -0.39] | 0.09 | test lower | exploratory |
| lora_baseline | lora_hn_relrep | replace_att | 2 | -1.59 | [-3.11, -0.06] | 0.99 | test lower | exploratory |
| lora_baseline | lora_hn_relrep | replace_rel | 2 | +6.76 | [+5.19, +8.36] | 0.20 | test higher | exploratory |
| lora_baseline | lora_hn_relrep | swap_obj | 2 | +0.82 | [-3.06, +4.69] | 1.15 | no detectable difference | exploratory |
| lora_baseline | lora_hn_relrep | swap_att | 2 | -2.25 | [-4.28, -0.23] | 1.06 | test lower | exploratory |
| lora_baseline | lora_hn_relrep | add_obj | 2 | -2.16 | [-2.96, -1.41] | 0.93 | test lower | exploratory |
| lora_baseline | lora_hn_relrep | add_att | 2 | +0.79 | [-1.16, +2.75] | 0.10 | no detectable difference | exploratory |
| lora_hn | lora_hn_objswap | replace_obj | 2 | -0.51 | [-1.18, +0.15] | 0.13 | no detectable difference | exploratory |
| lora_hn | lora_hn_objswap | replace_att | 2 | -1.52 | [-3.17, +0.13] | 0.18 | no detectable difference | exploratory |
| lora_hn | lora_hn_objswap | replace_rel | 2 | -5.58 | [-7.15, -4.05] | 1.46 | test lower | exploratory |
| lora_hn | lora_hn_objswap | swap_obj | 2 | +1.84 | [-1.22, +5.10] | 3.17 | no detectable difference | exploratory |
| lora_hn | lora_hn_objswap | swap_att | 2 | -1.35 | [-3.60, +0.90] | 1.06 | no detectable difference | exploratory |
| lora_hn | lora_hn_objswap | add_obj | 2 | -1.67 | [-2.50, -0.87] | 0.31 | test lower | exploratory |
| lora_hn | lora_hn_objswap | add_att | 2 | +0.87 | [-0.94, +2.67] | 1.43 | no detectable difference | exploratory |
| lora_hn | lora_hn_relrep | replace_obj | 2 | -0.18 | [-0.91, +0.51] | 0.09 | no detectable difference | exploratory |
| lora_hn | lora_hn_relrep | replace_att | 2 | -2.86 | [-4.70, -1.08] | 0.45 | test lower | exploratory |
| lora_hn | lora_hn_relrep | replace_rel | 2 | +0.50 | [-0.96, +1.96] | 0.30 | no detectable difference | exploratory |
| lora_hn | lora_hn_relrep | swap_obj | 2 | -3.88 | [-9.18, +1.43] | 0.87 | no detectable difference | exploratory |
| lora_hn | lora_hn_relrep | swap_att | 2 | -4.95 | [-7.51, -2.48] | 0.00 | test lower | exploratory |
| lora_hn | lora_hn_relrep | add_obj | 2 | -1.19 | [-2.11, -0.27] | 0.99 | test lower | exploratory |
| lora_hn | lora_hn_relrep | add_att | 2 | -3.54 | [-5.64, -1.45] | 0.51 | test lower | exploratory |
