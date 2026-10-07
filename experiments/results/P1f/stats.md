# P1f statistics (artefact checks)

Paired by seed (seeds 0-19, identical observation noise across arms); each seed's value is averaged over the 3x3 gamma x c0 grid (and seer set). diff = A - B with 95% bootstrap CI (10k), d_z, Wilcoxon signed-rank p, Holm across all 117 rows. 'better' says which sign is good for A. Metrics: log_score = mean log P(true sign of M) of agents' final beliefs (0 is perfect); brier = mean (1-P(true))^2; frac_agents_reach = share of agents with P(true) >= .75; community_reach = community mean >= .75; t_reach = rounds after the shift until the community mean first reaches .75 (censored at the horizon); track_brier = time-averaged Brier of the community against the current sign of M; abs_err_M = |posterior mean of M - 1| at the end.

## Check 1: outcome-based scoring

| contrast (A - B) | metric (better) | mean A | mean B | diff [95% CI] | d_z | p | p_Holm | Holm |
|---|---|---|---|---|---|---|---|---|
| pluralist - neutral monist | frac_agents_reach (higher) | 0.462 | 0.120 | +0.342 [+0.309, +0.375] | 4.43 | 8.8e-05 | 0.0041 | yes |
| pluralist - committed monist | frac_agents_reach (higher) | 0.462 | 0.006 | +0.456 [+0.422, +0.488] | 5.87 | 8.8e-05 | 0.0041 | yes |
| neutral - committed monist | frac_agents_reach (higher) | 0.120 | 0.006 | +0.114 [+0.112, +0.115] | 35.77 | 6.2e-05 | 0.0033 | yes |
| pluralist - neutral monist | community_reach (higher) | 0.478 | 0.111 | +0.367 [+0.333, +0.400] | 4.50 | 6.5e-05 | 0.0033 | yes |
| pluralist - committed monist | community_reach (higher) | 0.478 | 0.000 | +0.478 [+0.444, +0.511] | 5.87 | 6.5e-05 | 0.0033 | yes |
| neutral - committed monist | community_reach (higher) | 0.111 | 0.000 | +0.111 [+0.111, +0.111] | 7803672778549069.00 | 7.7e-06 | 0.00046 | yes |
| pluralist - neutral monist | log_score (higher) | -1.052 | -0.515 | -0.537 [-0.558, -0.519] | -11.69 | 1.9e-06 | 0.00022 | yes |
| pluralist - committed monist | log_score (higher) | -1.052 | -1.839 | +0.787 [+0.757, +0.814] | 11.80 | 1.9e-06 | 0.00022 | yes |
| neutral - committed monist | log_score (higher) | -0.515 | -1.839 | +1.324 [+1.295, +1.355] | 18.91 | 1.9e-06 | 0.00022 | yes |
| pluralist - neutral monist | brier (lower) | 0.366 | 0.166 | +0.199 [+0.193, +0.207] | 11.31 | 1.9e-06 | 0.00022 | yes |
| pluralist - committed monist | brier (lower) | 0.366 | 0.684 | -0.318 [-0.327, -0.307] | -13.25 | 1.9e-06 | 0.00022 | yes |
| neutral - committed monist | brier (lower) | 0.166 | 0.684 | -0.517 [-0.525, -0.510] | -28.96 | 1.9e-06 | 0.00022 | yes |
| pluralist - neutral monist | t_reach (lower) | 434.844 | 465.583 | -30.739 [-34.961, -26.817] | -3.18 | 1.9e-06 | 0.00022 | yes |
| pluralist - committed monist | t_reach (lower) | 434.844 | 520.000 | -85.156 [-89.444, -81.122] | -8.76 | 8.8e-05 | 0.0041 | yes |
| neutral - committed monist | t_reach (lower) | 465.583 | 520.000 | -54.417 [-54.528, -54.300] | -201.94 | 8.3e-05 | 0.004 | yes |
| P1c 75% - 25% pluralists (monists 50:50) | frac_agents_reach (higher) | 0.217 | 0.069 | +0.148 [+0.132, +0.164] | 3.93 | 8.7e-05 | 0.0041 | yes |
| P1c pluralists at observers - none | frac_agents_reach (higher) | 0.099 | 0.031 | +0.068 [+0.054, +0.081] | 2.17 | 0.00013 | 0.0056 | yes |
| P1c 75% - 25% pluralists (monists 50:50) | log_score (higher) | -0.909 | -0.915 | +0.006 [-0.020, +0.030] | 0.10 | 0.57 | 1 | no |
| P1c pluralists at observers - none | log_score (higher) | -0.819 | -0.970 | +0.151 [+0.134, +0.168] | 3.79 | 1.9e-06 | 0.00022 | yes |
| P1d relative(k1) - fixed, mode b attention [all_pluralist] | frac_agents_reach (higher) | 0.661 | 0.548 | +0.112 [+0.104, +0.122] | 5.30 | 4.4e-05 | 0.0024 | yes |
| P1e relative(k30) - fixed, mode a [all_pluralist] | frac_agents_reach (higher) | 0.616 | 0.462 | +0.154 [+0.108, +0.201] | 1.39 | 1.3e-05 | 0.00073 | yes |
| P1d relative(k1) - fixed, mode b attention [all_pluralist] | log_score (higher) | -0.788 | -0.986 | +0.199 [+0.192, +0.207] | 10.84 | 1.9e-06 | 0.00022 | yes |
| P1e relative(k30) - fixed, mode a [all_pluralist] | log_score (higher) | -0.816 | -1.052 | +0.236 [+0.181, +0.291] | 1.83 | 5.7e-06 | 0.00035 | yes |
| P1d agree - fixed, mode b attention [all_pluralist] | log_score (higher) | -1.341 | -0.986 | -0.354 [-0.362, -0.344] | -16.47 | 1.9e-06 | 0.00022 | yes |
| P1d relative(k1) - fixed, mode b attention [half_pluralist_committed] | frac_agents_reach (higher) | 0.118 | 0.113 | +0.005 [+0.002, +0.009] | 0.63 | 0.018 | 0.47 | no |
| P1e relative(k30) - fixed, mode a [half_pluralist_committed] | frac_agents_reach (higher) | 0.333 | 0.100 | +0.233 [+0.205, +0.263] | 3.38 | 1.9e-06 | 0.00022 | yes |
| P1d relative(k1) - fixed, mode b attention [half_pluralist_committed] | log_score (higher) | -0.897 | -0.977 | +0.079 [+0.074, +0.085] | 6.23 | 1.9e-06 | 0.00022 | yes |
| P1e relative(k30) - fixed, mode a [half_pluralist_committed] | log_score (higher) | -0.567 | -1.091 | +0.524 [+0.463, +0.584] | 3.71 | 1.9e-06 | 0.00022 | yes |
| P1d agree - fixed, mode b attention [half_pluralist_committed] | log_score (higher) | -1.160 | -0.977 | -0.184 [-0.201, -0.166] | -4.45 | 1.9e-06 | 0.00022 | yes |
| P1d relative(k1) - fixed, mode b attention [all_committed] | frac_agents_reach (higher) | 0.029 | 0.029 | +0.000 [+0.000, +0.000] | 0.00 | 1 | 1 | no |
| P1e relative(k30) - fixed, mode a [all_committed] | frac_agents_reach (higher) | 0.006 | 0.006 | +0.000 [+0.000, +0.000] | 0.00 | 1 | 1 | no |
| P1d relative(k1) - fixed, mode b attention [all_committed] | log_score (higher) | -1.760 | -1.760 | +0.000 [+0.000, +0.000] | 2.63 | 3.8e-06 | 0.00025 | yes (negligible) |
| P1e relative(k30) - fixed, mode a [all_committed] | log_score (higher) | -1.839 | -1.839 | +0.000 [+0.000, +0.000] | 2.05 | 9.5e-06 | 0.00055 | yes (negligible) |
| P1d agree - fixed, mode b attention [all_committed] | log_score (higher) | -1.937 | -1.760 | -0.177 [-0.183, -0.171] | -11.71 | 1.9e-06 | 0.00022 | yes |

Pure-type means at omega 0.9 (3x3 grid, 20 seeds):

| type | end P(M>0) | agents reach | community reach | log score | Brier | t_reach (censored share) |
|---|---|---|---|---|---|---|
| pluralist|noforget | 0.524 | 0.462 | 0.478 | -1.052 | 0.366 | 435 (0.52) |
| pluralist|forget | 0.606 | 0.111 | 0.111 | -0.510 | 0.163 | 466 (0.89) |
| neutral|noforget | 0.605 | 0.120 | 0.111 | -0.515 | 0.166 | 466 (0.89) |
| committed|noforget | 0.184 | 0.006 | 0.000 | -1.839 | 0.684 | 520 (1.00) |

## Check 2: changing worlds

| contrast (A - B) | metric (better) | mean A | mean B | diff [95% CI] | d_z | p | p_Holm | Holm |
|---|---|---|---|---|---|---|---|---|
| D=1 - D=50 [multiflip, all_pluralist, relative, every 10th] | track_brier (lower) | 0.197 | 0.289 | -0.092 [-0.102, -0.083] | -4.34 | 1.9e-06 | 0.00022 | yes |
| D=1 - D=50 [multiflip, all_pluralist, fixed, every 10th] | track_brier (lower) | 0.237 | 0.329 | -0.092 [-0.103, -0.081] | -3.51 | 1.9e-06 | 0.00022 | yes |
| relative(k30) - fixed, D=10 [multiflip, all_pluralist, every 10th] | track_brier (lower) | 0.220 | 0.256 | -0.036 [-0.046, -0.026] | -1.55 | 9.5e-06 | 0.00055 | yes |
| D=1 - D=50 [multiflip, all_pluralist, relative, every round] | track_brier (lower) | 0.064 | 0.333 | -0.268 [-0.277, -0.261] | -14.48 | 1.9e-06 | 0.00022 | yes |
| D=1 - D=50 [multiflip, half_pluralist_committed, relative, every 10th] | track_brier (lower) | 0.198 | 0.253 | -0.055 [-0.066, -0.045] | -2.19 | 1.9e-06 | 0.00022 | yes |
| D=1 - D=50 [multiflip, half_pluralist_committed, fixed, every 10th] | track_brier (lower) | 0.178 | 0.250 | -0.072 [-0.077, -0.067] | -6.32 | 1.9e-06 | 0.00022 | yes |
| relative(k30) - fixed, D=10 [multiflip, half_pluralist_committed, every 10th] | track_brier (lower) | 0.205 | 0.189 | +0.016 [+0.002, +0.028] | 0.53 | 0.036 | 0.87 | no |
| D=1 - D=50 [multiflip, half_pluralist_committed, relative, every round] | track_brier (lower) | 0.058 | 0.329 | -0.271 [-0.278, -0.264] | -15.99 | 1.9e-06 | 0.00022 | yes |
| D=1 - D=50 [drift, all_pluralist, relative, every 10th] | track_brier (lower) | 0.264 | 0.300 | -0.036 [-0.045, -0.027] | -1.74 | 3.8e-06 | 0.00025 | yes |
| D=1 - D=50 [drift, all_pluralist, fixed, every 10th] | track_brier (lower) | 0.278 | 0.305 | -0.028 [-0.037, -0.020] | -1.34 | 1.9e-06 | 0.00022 | yes |
| relative(k30) - fixed, D=10 [drift, all_pluralist, every 10th] | track_brier (lower) | 0.274 | 0.284 | -0.010 [-0.016, -0.003] | -0.63 | 0.021 | 0.54 | no |
| D=1 - D=50 [drift, all_pluralist, relative, every round] | track_brier (lower) | 0.072 | 0.158 | -0.086 [-0.098, -0.075] | -3.28 | 1.9e-06 | 0.00022 | yes |
| D=1 - D=50 [drift, half_pluralist_committed, relative, every 10th] | track_brier (lower) | 0.205 | 0.234 | -0.029 [-0.041, -0.018] | -1.07 | 9.5e-06 | 0.00055 | yes |
| D=1 - D=50 [drift, half_pluralist_committed, fixed, every 10th] | track_brier (lower) | 0.194 | 0.226 | -0.032 [-0.037, -0.027] | -2.81 | 1.9e-06 | 0.00022 | yes |
| relative(k30) - fixed, D=10 [drift, half_pluralist_committed, every 10th] | track_brier (lower) | 0.208 | 0.198 | +0.010 [-0.009, +0.027] | 0.23 | 0.43 | 1 | no |
| D=1 - D=50 [drift, half_pluralist_committed, relative, every round] | track_brier (lower) | 0.068 | 0.147 | -0.080 [-0.090, -0.070] | -3.41 | 1.9e-06 | 0.00022 | yes |

## Check 3: wrong rival

| contrast (A - B) | metric (better) | mean A | mean B | diff [95% CI] | d_z | p | p_Holm | Holm |
|---|---|---|---|---|---|---|---|---|
| pluralist - committed [original] | log_score (higher) | -1.052 | -1.839 | +0.787 [+0.757, +0.815] | 11.80 | 1.9e-06 | 0.00022 | yes |
| pluralist - committed [original] | abs_err_M (lower) | 0.985 | 1.792 | -0.807 [-0.832, -0.778] | -12.48 | 1.9e-06 | 0.00022 | yes |
| pluralist - neutral [original] | log_score (higher) | -1.052 | -0.515 | -0.537 [-0.558, -0.519] | -11.69 | 1.9e-06 | 0.00022 | yes |
| pluralist - committed [rival_M+3] | log_score (higher) | -1.863 | -1.863 | -0.000 [-0.000, -0.000] | -0.95 | 0.00085 | 0.034 | yes (negligible) |
| pluralist - committed [rival_M+3] | abs_err_M (lower) | 1.796 | 1.796 | -0.000 [-0.000, +0.000] | -0.23 | 0.012 | 0.35 | no |
| pluralist - neutral [rival_M+3] | log_score (higher) | -1.863 | -0.325 | -1.538 [-1.567, -1.509] | -22.74 | 1.9e-06 | 0.00022 | yes |
| pluralist - committed [rival_M-3] | log_score (higher) | -1.863 | -1.852 | -0.012 [-0.012, -0.011] | -8.84 | 1.9e-06 | 0.00022 | yes |
| pluralist - committed [rival_M-3] | abs_err_M (lower) | 1.796 | 1.798 | -0.001 [-0.001, -0.001] | -3.32 | 1.9e-06 | 0.00022 | yes (negligible) |
| pluralist - neutral [rival_M-3] | log_score (higher) | -1.863 | -2.125 | +0.262 [+0.234, +0.290] | 4.00 | 1.9e-06 | 0.00022 | yes |
| pluralist - committed [K3_-1_+1_+3] | log_score (higher) | -1.052 | -1.839 | +0.787 [+0.757, +0.814] | 11.80 | 1.9e-06 | 0.00022 | yes |
| pluralist - committed [K3_-1_+1_+3] | abs_err_M (lower) | 0.985 | 1.792 | -0.807 [-0.833, -0.777] | -12.48 | 1.9e-06 | 0.00022 | yes |
| pluralist - neutral [K3_-1_+1_+3] | log_score (higher) | -1.052 | -0.297 | -0.755 [-0.776, -0.736] | -16.22 | 1.9e-06 | 0.00022 | yes |

## Check 4: anchoring strength omega

| contrast (A - B) | metric (better) | mean A | mean B | diff [95% CI] | d_z | p | p_Holm | Holm |
|---|---|---|---|---|---|---|---|---|
| pluralist - committed [omega 0.5] | log_score (higher) | -1.825 | -1.981 | +0.155 [+0.152, +0.158] | 22.58 | 1.9e-06 | 0.00022 | yes |
| pluralist - neutral [omega 0.5] | log_score (higher) | -1.825 | -0.654 | -1.172 [-1.197, -1.148] | -19.85 | 1.9e-06 | 0.00022 | yes |
| pluralist(forget) - committed [omega 0.5] | log_score (higher) | -0.533 | -1.981 | +1.447 [+1.419, +1.477] | 21.11 | 1.9e-06 | 0.00022 | yes |
| pluralist - committed [omega 0.8] | log_score (higher) | -1.608 | -1.942 | +0.335 [+0.317, +0.353] | 7.90 | 1.9e-06 | 0.00022 | yes |
| pluralist - neutral [omega 0.8] | log_score (higher) | -1.608 | -0.586 | -1.022 [-1.051, -0.992] | -14.74 | 1.9e-06 | 0.00022 | yes |
| pluralist(forget) - committed [omega 0.8] | log_score (higher) | -0.525 | -1.942 | +1.417 [+1.381, +1.455] | 16.52 | 1.9e-06 | 0.00022 | yes |
| pluralist - committed [omega 0.9] | log_score (higher) | -1.052 | -1.839 | +0.787 [+0.757, +0.814] | 11.80 | 1.9e-06 | 0.00022 | yes |
| pluralist - neutral [omega 0.9] | log_score (higher) | -1.052 | -0.515 | -0.537 [-0.558, -0.519] | -11.69 | 1.9e-06 | 0.00022 | yes |
| pluralist(forget) - committed [omega 0.9] | log_score (higher) | -0.510 | -1.839 | +1.329 [+1.297, +1.361] | 17.91 | 1.9e-06 | 0.00022 | yes |
| pluralist - committed [omega 0.97] | log_score (higher) | -0.904 | -1.548 | +0.644 [+0.634, +0.653] | 29.38 | 1.9e-06 | 0.00022 | yes |
| pluralist - neutral [omega 0.97] | log_score (higher) | -0.904 | -0.400 | -0.504 [-0.509, -0.499] | -42.54 | 1.9e-06 | 0.00022 | yes |
| pluralist(forget) - committed [omega 0.97] | log_score (higher) | -0.459 | -1.548 | +1.089 [+1.078, +1.101] | 40.81 | 1.9e-06 | 0.00022 | yes |
| pluralist - committed [omega 1.0] | log_score (higher) | -12.308 | -12.313 | +0.005 [+0.005, +0.006] | 3.97 | 1.9e-06 | 0.00022 | yes |
| pluralist - neutral [omega 1.0] | log_score (higher) | -12.308 | -12.308 | +0.000 [-0.000, +0.000] | 0.26 | 0.37 | 1 | no |
| pluralist(forget) - committed [omega 1.0] | log_score (higher) | -12.308 | -12.313 | +0.005 [+0.005, +0.006] | 3.98 | 1.9e-06 | 0.00022 | yes |

Anchor tracking: anchor_pM = P(M>0) implied by the agent's own reference Gaussians weighted by its final support; mean |dev| = mean over agents of |end P(M>0) - anchor_pM|.

| omega | type | end P(M>0) | anchor P(M>0) | mean abs dev | corr(end, anchor) over cells x seeds | log score |
|---|---|---|---|---|---|---|
| 0.5 | pluralist|noforget | 0.216 | 0.221 | 0.025 | 1.00 | -1.825 |
| 0.5 | pluralist|forget | 0.591 | 0.585 | 0.008 | 1.00 | -0.533 |
| 0.5 | neutral|noforget | 0.522 | 0.500 | 0.023 | n/a | -0.654 |
| 0.5 | committed|noforget | 0.143 | 0.144 | 0.027 | n/a | -1.981 |
| 0.8 | pluralist|noforget | 0.300 | 0.302 | 0.028 | 1.00 | -1.608 |
| 0.8 | pluralist|forget | 0.596 | 0.576 | 0.021 | 0.99 | -0.525 |
| 0.8 | neutral|noforget | 0.561 | 0.500 | 0.061 | n/a | -0.586 |
| 0.8 | committed|noforget | 0.154 | 0.144 | 0.041 | n/a | -1.942 |
| 0.9 | pluralist|noforget | 0.524 | 0.509 | 0.038 | 1.00 | -1.052 |
| 0.9 | pluralist|forget | 0.606 | 0.567 | 0.039 | 0.93 | -0.510 |
| 0.9 | neutral|noforget | 0.605 | 0.500 | 0.105 | n/a | -0.515 |
| 0.9 | committed|noforget | 0.184 | 0.144 | 0.065 | n/a | -1.839 |
| 0.97 | pluralist|noforget | 0.569 | 0.537 | 0.050 | 1.00 | -0.904 |
| 0.97 | pluralist|forget | 0.642 | 0.544 | 0.098 | 0.24 | -0.459 |
| 0.97 | neutral|noforget | 0.680 | 0.500 | 0.180 | n/a | -0.400 |
| 0.97 | committed|noforget | 0.295 | 0.144 | 0.172 | n/a | -1.548 |
| 1.0 | pluralist|noforget | 0.087 | 0.214 | 0.186 | 0.99 | -12.308 |
| 1.0 | pluralist|forget | 0.087 | 0.502 | 0.476 | 0.92 | -12.308 |
| 1.0 | neutral|noforget | 0.087 | 0.500 | 0.476 | n/a | -12.308 |
| 1.0 | committed|noforget | 0.083 | 0.144 | 0.195 | n/a | -12.313 |

## Check 5: sparse/noisy reveals x number of full observers

| contrast (A - B) | metric (better) | mean A | mean B | diff [95% CI] | d_z | p | p_Holm | Holm |
|---|---|---|---|---|---|---|---|---|
| relative(k30) - fixed [all_pluralist, trust+belief, every 10, noise 1.0x, 4 observers] | log_score (higher) | -0.072 | -0.081 | +0.009 [+0.002, +0.017] | 0.54 | 0.036 | 0.87 | no |
| relative(k30) - fixed [all_pluralist, trust+belief, every 10, noise 1.0x, 8 observers] | log_score (higher) | -0.069 | -0.080 | +0.011 [+0.006, +0.016] | 0.92 | 0.0012 | 0.044 | yes |
| relative(k30) - fixed [all_pluralist, trust+belief, every 10, noise 1.0x, 12 observers] | log_score (higher) | -0.070 | -0.082 | +0.011 [+0.007, +0.016] | 1.05 | 0.00032 | 0.014 | yes |
| trust gain at 12 - at 4 observers [all_pluralist, trust+belief, every 10, noise 1.0x] | log_score (higher) | 0.011 | 0.009 | +0.002 [-0.005, +0.009] | 0.14 | 0.39 | 1 | no |
| relative(k30) - fixed [all_pluralist, trust+belief, every 10, noise 3.0x, 4 observers] | log_score (higher) | -0.297 | -0.401 | +0.104 [+0.060, +0.154] | 0.94 | 1.9e-06 | 0.00022 | yes |
| relative(k30) - fixed [all_pluralist, trust+belief, every 10, noise 3.0x, 8 observers] | log_score (higher) | -0.357 | -0.469 | +0.111 [+0.070, +0.159] | 1.07 | 1.9e-06 | 0.00022 | yes |
| relative(k30) - fixed [all_pluralist, trust+belief, every 10, noise 3.0x, 12 observers] | log_score (higher) | -0.434 | -0.584 | +0.150 [+0.097, +0.215] | 1.08 | 1.9e-06 | 0.00022 | yes |
| trust gain at 12 - at 4 observers [all_pluralist, trust+belief, every 10, noise 3.0x] | log_score (higher) | 0.150 | 0.104 | +0.046 [-0.018, +0.116] | 0.29 | 0.29 | 1 | no |
| relative(k30) - fixed [all_pluralist, trust+belief, every 30, noise 1.0x, 4 observers] | log_score (higher) | -0.099 | -0.108 | +0.009 [+0.001, +0.017] | 0.51 | 0.048 | 1 | no |
| relative(k30) - fixed [all_pluralist, trust+belief, every 30, noise 1.0x, 8 observers] | log_score (higher) | -0.099 | -0.107 | +0.008 [+0.003, +0.012] | 0.72 | 0.0073 | 0.23 | no |
| relative(k30) - fixed [all_pluralist, trust+belief, every 30, noise 1.0x, 12 observers] | log_score (higher) | -0.099 | -0.120 | +0.021 [+0.014, +0.031] | 1.03 | 5.7e-06 | 0.00035 | yes |
| trust gain at 12 - at 4 observers [all_pluralist, trust+belief, every 30, noise 1.0x] | log_score (higher) | 0.021 | 0.009 | +0.012 [+0.004, +0.020] | 0.66 | 0.0064 | 0.21 | no |
| relative(k30) - fixed [all_pluralist, trust+belief, every 30, noise 3.0x, 4 observers] | log_score (higher) | -0.817 | -0.927 | +0.110 [+0.056, +0.155] | 0.96 | 0.00059 | 0.024 | yes |
| relative(k30) - fixed [all_pluralist, trust+belief, every 30, noise 3.0x, 8 observers] | log_score (higher) | -0.780 | -0.896 | +0.116 [+0.054, +0.164] | 0.89 | 0.00085 | 0.034 | yes |
| relative(k30) - fixed [all_pluralist, trust+belief, every 30, noise 3.0x, 12 observers] | log_score (higher) | -0.790 | -0.891 | +0.100 [+0.049, +0.138] | 0.95 | 0.00085 | 0.034 | yes |
| trust gain at 12 - at 4 observers [all_pluralist, trust+belief, every 30, noise 3.0x] | log_score (higher) | 0.100 | 0.110 | -0.009 [-0.046, +0.031] | -0.10 | 0.52 | 1 | no |
| relative(k30) - fixed [all_pluralist, trust, every 10, noise 1.0x, 4 observers] | log_score (higher) | -0.821 | -1.052 | +0.231 [+0.150, +0.296] | 1.35 | 6.3e-05 | 0.0033 | yes |
| relative(k30) - fixed [all_pluralist, trust, every 10, noise 1.0x, 8 observers] | log_score (higher) | -0.771 | -1.014 | +0.243 [+0.188, +0.281] | 2.18 | 3.8e-06 | 0.00025 | yes |
| relative(k30) - fixed [all_pluralist, trust, every 10, noise 1.0x, 12 observers] | log_score (higher) | -0.782 | -0.982 | +0.200 [+0.159, +0.235] | 2.26 | 5.7e-06 | 0.00035 | yes |
| trust gain at 12 - at 4 observers [all_pluralist, trust, every 10, noise 1.0x] | log_score (higher) | 0.200 | 0.231 | -0.031 [-0.075, +0.034] | -0.24 | 0.011 | 0.33 | no |
| relative(k30) - fixed [half_pluralist_committed, trust+belief, every 10, noise 1.0x, 4 observers] | log_score (higher) | -0.447 | -0.409 | -0.038 [-0.126, +0.044] | -0.19 | 0.6 | 1 | no |
| relative(k30) - fixed [half_pluralist_committed, trust+belief, every 10, noise 1.0x, 8 observers] | log_score (higher) | -0.467 | -0.412 | -0.055 [-0.129, +0.024] | -0.31 | 0.22 | 1 | no |
| relative(k30) - fixed [half_pluralist_committed, trust+belief, every 10, noise 1.0x, 12 observers] | log_score (higher) | -0.490 | -0.435 | -0.055 [-0.134, +0.026] | -0.29 | 0.13 | 1 | no |
| trust gain at 12 - at 4 observers [half_pluralist_committed, trust+belief, every 10, noise 1.0x] | log_score (higher) | -0.055 | -0.038 | -0.017 [-0.086, +0.048] | -0.11 | 0.73 | 1 | no |
| relative(k30) - fixed [half_pluralist_committed, trust+belief, every 10, noise 3.0x, 4 observers] | log_score (higher) | -0.625 | -0.734 | +0.110 [+0.038, +0.177] | 0.68 | 0.011 | 0.33 | no |
| relative(k30) - fixed [half_pluralist_committed, trust+belief, every 10, noise 3.0x, 8 observers] | log_score (higher) | -0.610 | -0.683 | +0.073 [+0.005, +0.135] | 0.49 | 0.04 | 0.88 | no |
| relative(k30) - fixed [half_pluralist_committed, trust+belief, every 10, noise 3.0x, 12 observers] | log_score (higher) | -0.613 | -0.662 | +0.048 [-0.001, +0.099] | 0.41 | 0.09 | 1 | no |
| trust gain at 12 - at 4 observers [half_pluralist_committed, trust+belief, every 10, noise 3.0x] | log_score (higher) | 0.048 | 0.110 | -0.061 [-0.092, -0.029] | -0.83 | 0.0023 | 0.079 | no |
| relative(k30) - fixed [half_pluralist_committed, trust+belief, every 30, noise 1.0x, 4 observers] | log_score (higher) | -0.596 | -0.595 | -0.001 [-0.127, +0.120] | -0.01 | 0.99 | 1 | no |
| relative(k30) - fixed [half_pluralist_committed, trust+belief, every 30, noise 1.0x, 8 observers] | log_score (higher) | -0.615 | -0.581 | -0.034 [-0.140, +0.079] | -0.13 | 0.47 | 1 | no |
| relative(k30) - fixed [half_pluralist_committed, trust+belief, every 30, noise 1.0x, 12 observers] | log_score (higher) | -0.609 | -0.586 | -0.023 [-0.115, +0.074] | -0.10 | 0.55 | 1 | no |
| trust gain at 12 - at 4 observers [half_pluralist_committed, trust+belief, every 30, noise 1.0x] | log_score (higher) | -0.023 | -0.001 | -0.022 [-0.099, +0.049] | -0.12 | 0.67 | 1 | no |
| relative(k30) - fixed [half_pluralist_committed, trust+belief, every 30, noise 3.0x, 4 observers] | log_score (higher) | -0.868 | -0.964 | +0.096 [-0.040, +0.242] | 0.29 | 0.5 | 1 | no |
| relative(k30) - fixed [half_pluralist_committed, trust+belief, every 30, noise 3.0x, 8 observers] | log_score (higher) | -0.823 | -0.853 | +0.030 [-0.090, +0.164] | 0.10 | 0.84 | 1 | no |
| relative(k30) - fixed [half_pluralist_committed, trust+belief, every 30, noise 3.0x, 12 observers] | log_score (higher) | -0.787 | -0.775 | -0.011 [-0.122, +0.103] | -0.04 | 0.76 | 1 | no |
| trust gain at 12 - at 4 observers [half_pluralist_committed, trust+belief, every 30, noise 3.0x] | log_score (higher) | -0.011 | 0.096 | -0.107 [-0.162, -0.052] | -0.82 | 0.00085 | 0.034 | yes |
| relative(k30) - fixed [half_pluralist_committed, trust, every 10, noise 1.0x, 4 observers] | log_score (higher) | -0.792 | -1.092 | +0.300 [+0.190, +0.407] | 1.18 | 8.2e-05 | 0.004 | yes |
| relative(k30) - fixed [half_pluralist_committed, trust, every 10, noise 1.0x, 8 observers] | log_score (higher) | -0.793 | -0.998 | +0.205 [+0.105, +0.306] | 0.87 | 0.0017 | 0.059 | no |
| relative(k30) - fixed [half_pluralist_committed, trust, every 10, noise 1.0x, 12 observers] | log_score (higher) | -0.720 | -0.875 | +0.155 [+0.044, +0.265] | 0.60 | 0.015 | 0.41 | no |
| trust gain at 12 - at 4 observers [half_pluralist_committed, trust, every 10, noise 1.0x] | log_score (higher) | 0.155 | 0.300 | -0.145 [-0.237, -0.049] | -0.66 | 0.012 | 0.35 | no |

