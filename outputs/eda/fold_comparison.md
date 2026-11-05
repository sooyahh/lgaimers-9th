# Fold Comparison (2019–2024)

- csv: `outputs\eda\fold_comparison.csv`
- sources: `train.csv` (target/count), `trackman_history.csv` (release/spin/location/pitch_type)

## Highlight: requested year trends

| feature | metric | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | Δ2023−2022 | jump? | regime_from_2023? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `control_success` | mean | 0.56467 | 0.532712 | 0.532762 | 0.52892 | 0.499957 | 0.486105 | -0.0289632 | True | True |
| `pitch_type_group__other_rate` | mean | 0.0200463 | 0.0182462 | 0.0187721 | 0.00963798 | 0.00497937 | 0.00565186 | -0.00465861 | True | False |
| `tagged_pitch_type__top_category_share` | mean | 0.521552 | 0.48002 | 0.481454 | 0.472401 | 0.407023 | 0.448472 | -0.0653777 | True | True |
| `pitch_type_group__offspeed_rate` | mean | 0.144798 | 0.148449 | 0.148605 | 0.216161 | 0.220971 | 0.201807 | 0.00481052 | False | False |
| `rel_height` | mean | 1.70521 | 1.69966 | 1.71332 | 1.70004 | 1.69461 | 1.70633 | -0.00543367 | False | False |
| `rel_height` | std | 0.262994 | 0.254858 | 0.24827 | 0.253859 | 0.255906 | 0.225585 | 0.00204739 | False | True |
| `rel_height` | missing_rate | 0.0044695 | 0.00396237 | 0.00512902 | 0.00442079 | 0.00318629 | 0.00436531 | -0.0012345 | False | False |
| `rel_side` | mean | 0.253559 | 0.286019 | 0.269129 | 0.279321 | 0.253993 | 0.242957 | -0.0253274 | True | True |
| `rel_side` | std | 0.538305 | 0.507888 | 0.54739 | 0.531735 | 0.529046 | 0.535027 | -0.00268895 | False | False |
| `rel_side` | missing_rate | 0.0044695 | 0.00396237 | 0.00512902 | 0.00442405 | 0.00318629 | 0.00436531 | -0.00123776 | False | False |
| `extension` | mean | 1.80347 | 1.83539 | 1.8483 | 1.70285 | 1.69501 | 1.6935 | -0.00783968 | False | True |
| `extension` | std | 0.150553 | 0.151095 | 0.151879 | 0.152156 | 0.15505 | 0.153893 | 0.00289391 | False | False |
| `extension` | missing_rate | 0.0044695 | 0.00396237 | 0.00512902 | 0.00452806 | 0.00330371 | 0.00445208 | -0.00122435 | False | False |
| `rel_speed` | mean | 135.466 | 135.565 | 135.615 | 136.433 | 136.186 | 136.352 | -0.246384 | False | False |
| `rel_speed` | std | 9.44145 | 9.38423 | 9.26743 | 9.45746 | 9.33351 | 9.31315 | -0.123946 | False | False |
| `rel_speed` | missing_rate | 0.0044695 | 0.00396237 | 0.00512902 | 0.00442079 | 0.00318629 | 0.00436531 | -0.0012345 | False | False |
| `spin_rate` | mean | 2169.26 | 2158.03 | 2170.9 | 2170.94 | 2176.14 | 2172.87 | 5.20017 | False | False |
| `spin_rate` | std | 342.217 | 334.047 | 327.882 | 328.559 | 338.616 | 354.681 | 10.0579 | False | False |
| `spin_rate` | missing_rate | 0.0111777 | 0.0102391 | 0.00964017 | 0.00445655 | 0.00321485 | 0.0043683 | -0.0012417 | False | True |
| `induced_vert_break` | mean | 27.6839 | 27.9336 | 27.4963 | 25.204 | 24.423 | 22.8017 | -0.780987 | True | True |
| `induced_vert_break` | std | 23.8734 | 23.6318 | 23.8037 | 25.1378 | 24.4719 | 24.4153 | -0.66589 | False | False |
| `induced_vert_break` | missing_rate | 0.0044695 | 0.00396237 | 0.00512902 | 0.00966399 | 0.00483656 | 0.00529881 | -0.00482743 | False | False |
| `horz_break` | mean | 8.68174 | 10.3742 | 9.71692 | 10.1298 | 9.46211 | 8.41348 | -0.667716 | True | True |
| `horz_break` | std | 25.0424 | 25.1148 | 25.7727 | 26.2059 | 26.5799 | 26.2701 | 0.374027 | False | False |
| `horz_break` | missing_rate | 0.00483284 | 0.00425614 | 0.00544793 | 0.00966399 | 0.00483656 | 0.00529881 | -0.00482743 | False | False |
| `zone_speed` | mean | 123.913 | 124.423 | 124.376 | 125.102 | 124.434 | 125.118 | -0.668521 | False | False |
| `zone_speed` | std | 8.45117 | 8.46517 | 8.31266 | 8.48426 | 8.25836 | 8.3038 | -0.225906 | False | False |
| `zone_speed` | missing_rate | 0.0044695 | 0.00396237 | 0.00613888 | 0.00442079 | 0.00318629 | 0.00436531 | -0.0012345 | False | False |

## Sample counts

| season | train n_rows | trackman n_rows |
|---:|---:|---:|
| 2019 | 237,413 | 255,957 |
| 2020 | 244,087 | 279,126 |
| 2021 | 247,088 | 301,032 |
| 2022 | 247,472 | 307,637 |
| 2023 | 245,525 | 315,100 |
| 2024 | 253,507 | 334,226 |

## 2022→2023 sharp changes

| feature | metric | 2022 | 2023 | Δ | |rel| |
|---|---|---:|---:|---:|---:|
| `pitch_type_group__other_rate` | mean | 0.00963798 | 0.00497937 | -0.00465861 | 0.4834 |
| `tagged_pitch_type__top_category_share` | mean | 0.472401 | 0.407023 | -0.0653777 | 0.1384 |
| `rel_side` | mean | 0.279321 | 0.253993 | -0.0253274 | 0.0907 |
| `horz_break` | mean | 10.1298 | 9.46211 | -0.667716 | 0.0659 |
| `control_success` | rate_0 | 0.47108 | 0.500043 | 0.0289632 | 0.0615 |
| `control_success` | mean | 0.52892 | 0.499957 | -0.0289632 | 0.0548 |
| `control_success` | rate_1 | 0.52892 | 0.499957 | -0.0289632 | 0.0548 |
| `induced_vert_break` | mean | 25.204 | 24.423 | -0.780987 | 0.0310 |
| `balls_before` | mean | 0.885458 | 0.901289 | 0.0158313 | 0.0179 |

## Stable 2019–2022, different from 2023+

Criterion: early years relatively stable (`early_cv` low) and mean(2023,2024) differs from mean(2019–2022) beyond thresholds.

| feature | metric | early_mean | late_mean | Δ | |rel| | early_cv |
|---|---|---:|---:|---:|---:|---:|
| `spin_rate` | missing_rate | 0.00887837 | 0.00379158 | -0.00508679 | 0.5729 | 0.2941 |
| `pitch_type_group__breaking_rate` | mean | 0.267228 | 0.317592 | 0.0503643 | 0.1885 | 0.0642 |
| `pitch_type_group__fastball_rate` | mean | 0.551593 | 0.465703 | -0.0858899 | 0.1557 | 0.0778 |
| `n_rows` | count | 285938 | 324663 | 38725 | 0.1354 | 0.0709 |
| `induced_vert_break` | mean | 27.0795 | 23.6123 | -3.46714 | 0.1280 | 0.0404 |
| `tagged_pitch_type__top_category_share` | mean | 0.488857 | 0.427748 | -0.0611092 | 0.1250 | 0.0393 |
| `control_success` | rate_0 | 0.460234 | 0.506969 | 0.0467349 | 0.1015 | 0.0314 |
| `control_success` | mean | 0.539766 | 0.493031 | -0.0467349 | 0.0866 | 0.0268 |
| `control_success` | rate_1 | 0.539766 | 0.493031 | -0.0467349 | 0.0866 | 0.0268 |
| `rel_side` | mean | 0.272007 | 0.248475 | -0.0235319 | 0.0865 | 0.0450 |
| `horz_break` | mean | 9.72568 | 8.9378 | -0.787885 | 0.0810 | 0.0665 |
| `extension` | mean | 1.7975 | 1.69426 | -0.103246 | 0.0574 | 0.0317 |
| `rel_height` | std | 0.254995 | 0.240745 | -0.0142498 | 0.0559 | 0.0206 |

