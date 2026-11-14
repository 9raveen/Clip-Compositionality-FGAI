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
| lora_hn_rel | 1 | 29.5 | 12.0 | 7.8 | 71.7 | 92.3 | 96.6 | 59.6 | 85.8 | 93.6 | 0.406 |

val L_CLIP is comparable across conditions. Validation hn_acc is NOT in this table: it is measured on
different negative files for different conditions and can be inflated by text-only shortcuts.

## Paired bootstrap on Winoground (difference = test − ref, percentage points)

| ref | test | metric | seeds | diff | 95% CI | seed std of diff | verdict |
|---|---|---|---|---|---|---|---|
| lora_baseline | lora_hn | text | 3 | -4.50 | [-7.75, -1.33] | 1.32 | test lower |
| lora_baseline | lora_hn | image | 3 | -0.42 | [-2.33, +1.50] | 2.25 | no detectable difference |
| lora_baseline | lora_hn | group | 3 | -0.67 | [-2.50, +1.08] | 1.04 | no detectable difference |
| lora_baseline | lora_hn_plaus | text | 3 | -0.58 | [-2.92, +1.67] | 2.25 | no detectable difference |
| lora_baseline | lora_hn_plaus | image | 3 | -0.33 | [-2.33, +1.58] | 1.94 | no detectable difference |
| lora_baseline | lora_hn_plaus | group | 3 | +0.17 | [-1.50, +1.75] | 0.76 | no detectable difference |
| lora_hn_plaus | lora_hn | text | 3 | -3.92 | [-6.67, -1.25] | 1.15 | test lower |
| lora_hn_plaus | lora_hn | image | 3 | -0.08 | [-1.42, +1.33] | 0.52 | no detectable difference |
| lora_hn_plaus | lora_hn | group | 3 | -0.83 | [-2.08, +0.42] | 0.29 | no detectable difference |
| lora_baseline | lora_hn_attr | text | 1 | -0.25 | [-3.75, +3.25] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_attr | image | 1 | -3.00 | [-5.75, -0.50] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_attr | group | 1 | -1.50 | [-3.50, +0.50] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_obj | text | 1 | +0.50 | [-2.50, +3.50] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_obj | image | 1 | -3.00 | [-5.25, -0.75] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_obj | group | 1 | -1.00 | [-3.00, +1.00] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_rel | text | 1 | -4.00 | [-8.25, +0.00] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_rel | image | 1 | -2.25 | [-4.75, +0.00] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_rel | group | 1 | -1.25 | [-3.50, +1.00] | 0.00 | no detectable difference |
| lora_hn | lora_hn_attr | text | 1 | +3.75 | [-0.25, +7.75] | 0.00 | no detectable difference |
| lora_hn | lora_hn_attr | image | 1 | -0.25 | [-2.75, +2.25] | 0.00 | no detectable difference |
| lora_hn | lora_hn_attr | group | 1 | +0.00 | [-2.00, +2.00] | 0.00 | no detectable difference |
| lora_hn | lora_hn_obj | text | 1 | +4.50 | [+0.75, +8.25] | 0.00 | test higher |
| lora_hn | lora_hn_obj | image | 1 | -0.25 | [-2.50, +2.00] | 0.00 | no detectable difference |
| lora_hn | lora_hn_obj | group | 1 | +0.50 | [-1.50, +2.50] | 0.00 | no detectable difference |
| lora_hn | lora_hn_rel | text | 1 | +0.00 | [-3.00, +3.00] | 0.00 | no detectable difference |
| lora_hn | lora_hn_rel | image | 1 | +0.50 | [-1.50, +2.50] | 0.00 | no detectable difference |
| lora_hn | lora_hn_rel | group | 1 | +0.25 | [-1.50, +2.00] | 0.00 | no detectable difference |

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
| lora_baseline | lora_hn_rel | i2t_r1 | -3.10 | nan |
| lora_baseline | lora_hn_rel | t2i_r1 | -0.60 | nan |
| lora_hn | lora_hn_attr | i2t_r1 | +0.10 | nan |
| lora_hn | lora_hn_attr | t2i_r1 | -0.64 | nan |
| lora_hn | lora_hn_obj | i2t_r1 | +0.20 | nan |
| lora_hn | lora_hn_obj | t2i_r1 | -0.44 | nan |
| lora_hn | lora_hn_rel | i2t_r1 | -1.40 | nan |
| lora_hn | lora_hn_rel | t2i_r1 | -0.32 | nan |

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
| lora_hn_rel | 61.5 | 34.6 | 26.9 | 36.2 | 11.3 | 7.8 | 21.9 | 9.9 | 5.6 |
| n examples | 26 | 26 | 26 | 141 | 141 | 141 | 233 | 233 | 233 |

## Paired bootstrap by tag (diff = test − ref, pp). Tags with n < 30 are listed but must not be interpreted.

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
| lora_baseline | lora_hn_rel | Both (26) | text | 1 | -3.85 | [-15.38, +7.69] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_rel | Both (26) | image | 1 | -3.85 | [-11.54, +0.00] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_rel | Both (26) | group | 1 | -3.85 | [-11.54, +0.00] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_rel | Object (141) | text | 1 | -7.80 | [-15.60, +0.00] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_rel | Object (141) | image | 1 | -5.67 | [-10.64, -1.42] | 0.00 | test lower |
| lora_baseline | lora_hn_rel | Object (141) | group | 1 | -4.26 | [-8.51, +0.00] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_rel | Relation (233) | text | 1 | -1.72 | [-6.44, +3.00] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_rel | Relation (233) | image | 1 | +0.00 | [-3.00, +3.00] | 0.00 | no detectable difference |
| lora_baseline | lora_hn_rel | Relation (233) | group | 1 | +0.86 | [-1.29, +3.43] | 0.00 | no detectable difference |
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
| lora_hn | lora_hn_rel | Both (26) | text | 1 | -3.85 | [-11.54, +0.00] | 0.00 | no detectable difference |
| lora_hn | lora_hn_rel | Both (26) | image | 1 | +3.85 | [+0.00, +11.54] | 0.00 | no detectable difference |
| lora_hn | lora_hn_rel | Both (26) | group | 1 | +0.00 | [+0.00, +0.00] | 0.00 | no detectable difference |
| lora_hn | lora_hn_rel | Object (141) | text | 1 | +1.42 | [-4.26, +7.09] | 0.00 | no detectable difference |
| lora_hn | lora_hn_rel | Object (141) | image | 1 | -0.71 | [-3.55, +2.13] | 0.00 | no detectable difference |
| lora_hn | lora_hn_rel | Object (141) | group | 1 | +0.00 | [-3.55, +3.55] | 0.00 | no detectable difference |
| lora_hn | lora_hn_rel | Relation (233) | text | 1 | -0.43 | [-4.29, +3.43] | 0.00 | no detectable difference |
| lora_hn | lora_hn_rel | Relation (233) | image | 1 | +0.86 | [-2.15, +3.86] | 0.00 | no detectable difference |
| lora_hn | lora_hn_rel | Relation (233) | group | 1 | +0.43 | [-1.72, +2.58] | 0.00 | no detectable difference |
