# P1g statistics: population size, message bandwidth, neutral drift

Part 1 contrasts are unpaired (different N = different random draws): difference of seed-level means (cell-averaged), independent bootstrap CI (10k), Mann-Whitney U. The lottery indicator is the across-seed SD of the community-mean end P(M>0) within a cell, averaged over the 3x3 grid. Part 2 is paired by seed (Wilcoxon). Holm across all 64 tests.

| part | contrast (A - B) | metric | mean A | mean B | diff [95% CI] | p | p_Holm | Holm |
|---|---|---|---|---|---|---|---|---|
| 1 | N=100 - N=8 [pluralist, observers sixth] | end_pM | 0.532 | 0.485 | +0.048 [+0.028, +0.067] | 0.0098 | 0.28 | no |
| 1 | N=100 - N=8 [pluralist, observers sixth] | frac_agents_reach | 0.509 | 0.425 | +0.084 [+0.041, +0.126] | 0.013 | 0.36 | no |
| 1 | N=100 - N=8 [pluralist, observers sixth] | log_score | -1.035 | -1.150 | +0.116 [+0.070, +0.157] | 0.001 | 0.032 | yes |
| 1 | N=100 - N=8 [pluralist, observers sixth] | t_reach | 431.489 | 448.372 | -16.883 [-22.711, -11.172] | 7.4e-05 | 0.0031 | yes |
| 1 | N=100 - N=8 [pluralist, observers sixth] | across-seed SD of end P(M>0) (lottery) | 0.011 | 0.057 | -0.046 [-0.051, -0.036] | 0.0005 | 0.021 | yes |
| 1 | N=100 - N=8 [committed, observers sixth] | end_pM | 0.187 | 0.169 | +0.018 [+0.010, +0.025] | 3.7e-05 | 0.0016 | yes |
| 1 | N=100 - N=8 [committed, observers sixth] | frac_agents_reach | 0.007 | 0.003 | +0.003 [+0.001, +0.006] | 0.0055 | 0.17 | no |
| 1 | N=100 - N=8 [committed, observers sixth] | log_score | -1.821 | -1.892 | +0.071 [+0.025, +0.119] | 0.0043 | 0.13 | no |
| 1 | N=100 - N=8 [committed, observers sixth] | t_reach | 520.000 | 520.000 | +0.000 [+0.000, +0.000] | 1 | 1 | no |
| 1 | N=100 - N=8 [committed, observers sixth] | across-seed SD of end P(M>0) (lottery) | 0.007 | 0.018 | -0.012 [-0.015, -0.007] | 0.0005 | 0.021 | yes |
| 1 | N=100 - N=8 [neutral, observers sixth] | end_pM | 0.606 | 0.592 | +0.015 [+0.011, +0.018] | 1.8e-06 | 8.3e-05 | yes |
| 1 | N=100 - N=8 [neutral, observers sixth] | frac_agents_reach | 0.119 | 0.110 | +0.009 [-0.001, +0.024] | 1 | 1 | no |
| 1 | N=100 - N=8 [neutral, observers sixth] | log_score | -0.513 | -0.535 | +0.022 [+0.016, +0.028] | 1e-06 | 5e-05 | yes |
| 1 | N=100 - N=8 [neutral, observers sixth] | t_reach | 465.444 | 466.300 | -0.856 [-1.167, -0.578] | 1.1e-06 | 5.3e-05 | yes |
| 1 | N=100 - N=8 [neutral, observers sixth] | across-seed SD of end P(M>0) (lottery) | 0.004 | 0.010 | -0.006 [-0.008, -0.004] | 0.0005 | 0.021 | yes |
| 1 | N=100 - N=8 [mix, observers sixth] | end_pM | 0.384 | 0.418 | -0.034 [-0.054, -0.016] | 0.00051 | 0.021 | yes |
| 1 | N=100 - N=8 [mix, observers sixth] | frac_agents_reach | 0.116 | 0.122 | -0.005 [-0.015, +0.003] | 0.077 | 1 | no |
| 1 | N=100 - N=8 [mix, observers sixth] | log_score | -1.078 | -1.022 | -0.056 [-0.118, +0.004] | 0.024 | 0.65 | no |
| 1 | N=100 - N=8 [mix, observers sixth] | t_reach | 480.961 | 474.378 | +6.583 [+4.711, +8.400] | 3e-06 | 0.00013 | yes |
| 1 | N=100 - N=8 [mix, observers sixth] | across-seed SD of end P(M>0) (lottery) | 0.011 | 0.048 | -0.038 [-0.055, -0.014] | 0.0005 | 0.021 | yes |
| 1 | N=100 - N=8 [pluralist, observers four] | end_pM | 0.372 | 0.544 | -0.172 [-0.176, -0.167] | 6.8e-08 | 3.9e-06 | yes |
| 1 | N=100 - N=8 [pluralist, observers four] | frac_agents_reach | 0.325 | 0.459 | -0.134 [-0.149, -0.121] | 2.9e-08 | 1.7e-06 | yes |
| 1 | N=100 - N=8 [pluralist, observers four] | log_score | -1.410 | -0.997 | -0.413 [-0.432, -0.395] | 6.8e-08 | 3.9e-06 | yes |
| 1 | N=100 - N=8 [pluralist, observers four] | t_reach | 480.983 | 372.572 | +108.411 [+106.372, +110.433] | 6.8e-08 | 3.9e-06 | yes |
| 1 | N=100 - N=8 [pluralist, observers four] | across-seed SD of end P(M>0) (lottery) | 0.008 | 0.018 | -0.010 [-0.013, -0.006] | 0.0005 | 0.021 | yes |
| 1 | N=100 - N=8 [committed, observers four] | end_pM | 0.141 | 0.272 | -0.130 [-0.137, -0.124] | 6.8e-08 | 3.9e-06 | yes |
| 1 | N=100 - N=8 [committed, observers four] | frac_agents_reach | 0.000 | 0.117 | -0.117 [-0.122, -0.113] | 2.2e-08 | 1.3e-06 | yes |
| 1 | N=100 - N=8 [committed, observers four] | log_score | -1.990 | -1.651 | -0.339 [-0.376, -0.299] | 6.8e-08 | 3.9e-06 | yes |
| 1 | N=100 - N=8 [committed, observers four] | t_reach | 520.000 | 464.628 | +55.372 [+55.200, +55.533] | 7.6e-09 | 4.8e-07 | yes |
| 1 | N=100 - N=8 [committed, observers four] | across-seed SD of end P(M>0) (lottery) | 0.005 | 0.018 | -0.014 [-0.017, -0.009] | 0.0005 | 0.021 | yes |
| 1 | N=100 - N=8 [neutral, observers four] | end_pM | 0.562 | 0.640 | -0.078 [-0.081, -0.074] | 6.8e-08 | 3.9e-06 | yes |
| 1 | N=100 - N=8 [neutral, observers four] | frac_agents_reach | 0.006 | 0.188 | -0.182 [-0.198, -0.166] | 5.2e-08 | 3e-06 | yes |
| 1 | N=100 - N=8 [neutral, observers four] | log_score | -0.580 | -0.467 | -0.113 [-0.119, -0.107] | 6.8e-08 | 3.9e-06 | yes |
| 1 | N=100 - N=8 [neutral, observers four] | t_reach | 520.000 | 406.906 | +113.094 [+110.161, +118.800] | 7.8e-09 | 4.9e-07 | yes |
| 1 | N=100 - N=8 [neutral, observers four] | across-seed SD of end P(M>0) (lottery) | 0.004 | 0.009 | -0.006 [-0.008, -0.003] | 0.0005 | 0.021 | yes |
| 1 | N=100 - N=8 [mix, observers four] | end_pM | 0.323 | 0.475 | -0.153 [-0.176, -0.128] | 6.8e-08 | 3.9e-06 | yes |
| 1 | N=100 - N=8 [mix, observers four] | frac_agents_reach | 0.004 | 0.163 | -0.159 [-0.169, -0.149] | 5.9e-08 | 3.4e-06 | yes |
| 1 | N=100 - N=8 [mix, observers four] | log_score | -1.212 | -0.856 | -0.356 [-0.425, -0.285] | 2.2e-07 | 1.1e-05 | yes |
| 1 | N=100 - N=8 [mix, observers four] | t_reach | 520.000 | 431.850 | +88.150 [+85.328, +91.017] | 8e-09 | 5e-07 | yes |
| 1 | N=100 - N=8 [mix, observers four] | across-seed SD of end P(M>0) (lottery) | 0.010 | 0.060 | -0.050 [-0.058, -0.039] | 0.0005 | 0.021 | yes |
| 2 | m=1 - exact moments [pluralist] | end_pM | 0.558 | 0.583 | -0.025 [-0.092, +0.040] | 0.52 | 1 | no |
| 2 | m=1 - exact moments [pluralist] | log_score | -0.644 | -0.549 | -0.094 [-0.239, +0.040] | 0.45 | 1 | no |
| 2 | m=1 - exact moments [pluralist] | frac_agents_reach | 0.156 | 0.110 | +0.046 [-0.038, +0.146] | 0.98 | 1 | no |
| 2 | exact observable moments - full natural parameters [pluralist] | end_pM | 0.583 | 0.524 | +0.060 [+0.052, +0.069] | 1.9e-06 | 8.6e-05 | yes |
| 2 | sampled support m=1 - exact support (message m=1) [pluralist] | end_pM | 0.560 | 0.558 | +0.002 [-0.000, +0.005] | 0.26 | 1 | no |
| 2 | sampled support m=1 - exact support (message m=1) [pluralist] | log_score | -0.639 | -0.644 | +0.005 [+0.000, +0.011] | 0.26 | 1 | no |
| 2 | m=1 - exact moments [committed] | end_pM | 0.192 | 0.184 | +0.008 [-0.038, +0.059] | 0.96 | 1 | no |
| 2 | m=1 - exact moments [committed] | log_score | -2.011 | -1.839 | -0.173 [-0.489, +0.131] | 0.52 | 1 | no |
| 2 | m=1 - exact moments [committed] | frac_agents_reach | 0.022 | 0.006 | +0.016 [+0.001, +0.033] | 0.13 | 1 | no |
| 2 | exact observable moments - full natural parameters [committed] | end_pM | 0.184 | 0.184 | +0.000 [+0.000, +0.000] | 1 | 1 | no |
| 2 | m=1 - exact moments [neutral] | end_pM | 0.574 | 0.605 | -0.031 [-0.100, +0.036] | 0.45 | 1 | no |
| 2 | m=1 - exact moments [neutral] | log_score | -0.623 | -0.515 | -0.108 [-0.257, +0.029] | 0.25 | 1 | no |
| 2 | m=1 - exact moments [neutral] | frac_agents_reach | 0.250 | 0.120 | +0.130 [+0.005, +0.283] | 0.42 | 1 | no |
| 2 | exact observable moments - full natural parameters [neutral] | end_pM | 0.605 | 0.605 | -0.000 [-0.000, +0.000] | 1 | 1 | no |
| 2 | m=1 - exact moments [mix] | end_pM | 0.374 | 0.382 | -0.008 [-0.072, +0.056] | 0.76 | 1 | no |
| 2 | m=1 - exact moments [mix] | log_score | -1.135 | -1.015 | -0.120 [-0.348, +0.090] | 0.62 | 1 | no |
| 2 | m=1 - exact moments [mix] | frac_agents_reach | 0.041 | 0.020 | +0.021 [+0.001, +0.045] | 0.68 | 1 | no |
| 2 | exact observable moments - full natural parameters [mix] | end_pM | 0.382 | 0.379 | +0.003 [-0.004, +0.010] | 0.5 | 1 | no |
| 2 | sampled support m=1 - exact support (message m=1) [mix] | end_pM | 0.376 | 0.374 | +0.001 [+0.000, +0.003] | 0.07 | 1 | no |
| 2 | sampled support m=1 - exact support (message m=1) [mix] | log_score | -1.129 | -1.135 | +0.006 [+0.001, +0.012] | 0.064 | 1 | no |
| 2 | bandwidth loss: pluralist - committed [(m=1) - (exact)] | end_pM | -0.025 | 0.008 | -0.033 [-0.064, -0.006] | 0.07 | 1 | no |
| 2 | bandwidth loss: pluralist - neutral [(m=1) - (exact)] | end_pM | -0.025 | -0.031 | +0.006 [-0.001, +0.014] | 0.15 | 1 | no |
| 2 | bandwidth loss: pluralist - committed [(m=1) - (exact)] | log_score | -0.094 | -0.173 | +0.078 [-0.108, +0.259] | 0.5 | 1 | no |
| 2 | bandwidth loss: pluralist - neutral [(m=1) - (exact)] | log_score | -0.094 | -0.108 | +0.013 [-0.003, +0.029] | 0.058 | 1 | no |

## Part 1: population size (means over 3x3 grid x 20 seeds)

| arm | end P(M>0) | across-seed SD | share of seeds locked | p(1-p) bimodality | agents reach | t_reach |
|---|---|---|---|---|---|---|
| N100|four|committed | 0.141 | 0.005 | 1.00 | 0.000 | 0.00 | 520 |
| N100|four|mix | 0.323 | 0.010 | 0.80 | 0.018 | 0.00 | 520 |
| N100|four|neutral | 0.562 | 0.004 | 0.00 | 0.000 | 0.01 | 520 |
| N100|four|pluralist | 0.372 | 0.008 | 0.67 | 0.000 | 0.33 | 481 |
| N100|sixth|committed | 0.187 | 0.007 | 0.89 | 0.000 | 0.01 | 520 |
| N100|sixth|mix | 0.384 | 0.011 | 0.78 | 0.000 | 0.12 | 481 |
| N100|sixth|neutral | 0.606 | 0.004 | 0.00 | 0.000 | 0.12 | 465 |
| N100|sixth|pluralist | 0.532 | 0.011 | 0.44 | 0.000 | 0.51 | 431 |
| N24|sixth|committed | 0.184 | 0.016 | 0.91 | 0.018 | 0.01 | 520 |
| N24|sixth|mix | 0.379 | 0.022 | 0.78 | 0.000 | 0.10 | 480 |
| N24|sixth|neutral | 0.605 | 0.008 | 0.00 | 0.000 | 0.12 | 466 |
| N24|sixth|pluralist | 0.524 | 0.028 | 0.46 | 0.010 | 0.46 | 435 |
| N50|four|committed | 0.156 | 0.008 | 1.00 | 0.000 | 0.00 | 520 |
| N50|four|mix | 0.350 | 0.011 | 0.78 | 0.000 | 0.01 | 520 |
| N50|four|neutral | 0.578 | 0.006 | 0.00 | 0.000 | 0.05 | 469 |
| N50|four|pluralist | 0.461 | 0.032 | 0.51 | 0.027 | 0.32 | 473 |
| N50|sixth|committed | 0.184 | 0.010 | 0.92 | 0.023 | 0.01 | 520 |
| N50|sixth|mix | 0.379 | 0.010 | 0.78 | 0.000 | 0.10 | 484 |
| N50|sixth|neutral | 0.604 | 0.006 | 0.00 | 0.000 | 0.12 | 466 |
| N50|sixth|pluralist | 0.527 | 0.024 | 0.46 | 0.010 | 0.49 | 435 |
| N8|four|committed | 0.272 | 0.018 | 0.78 | 0.000 | 0.12 | 465 |
| N8|four|mix | 0.475 | 0.060 | 0.72 | 0.033 | 0.16 | 432 |
| N8|four|neutral | 0.640 | 0.009 | 0.00 | 0.000 | 0.19 | 407 |
| N8|four|pluralist | 0.544 | 0.018 | 0.44 | 0.000 | 0.46 | 373 |
| N8|sixth|committed | 0.169 | 0.018 | 0.98 | 0.014 | 0.00 | 520 |
| N8|sixth|mix | 0.418 | 0.048 | 0.71 | 0.038 | 0.12 | 474 |
| N8|sixth|neutral | 0.592 | 0.010 | 0.00 | 0.000 | 0.11 | 466 |
| N8|sixth|pluralist | 0.485 | 0.057 | 0.51 | 0.027 | 0.42 | 448 |

## Part 2: bandwidth (N=24, means over 3x3 grid x 20 seeds)

| arm (type, fusion, message m, support m) | end P(M>0) | log score | agents reach |
|---|---|---|---|
| pluralist|rule|msg0|sup0 | 0.524 | -1.052 | 0.46 |
| pluralist|observables|msg0|sup0 | 0.583 | -0.549 | 0.11 |
| pluralist|observables|msg10|sup0 | 0.574 | -0.570 | 0.09 |
| pluralist|observables|msg3|sup0 | 0.567 | -0.591 | 0.10 |
| pluralist|observables|msg1|sup0 | 0.558 | -0.644 | 0.16 |
| pluralist|observables|msg10|sup10 | 0.573 | -0.571 | 0.09 |
| pluralist|observables|msg3|sup3 | 0.571 | -0.583 | 0.11 |
| pluralist|observables|msg1|sup1 | 0.560 | -0.639 | 0.16 |
| committed|rule|msg0|sup0 | 0.184 | -1.839 | 0.01 |
| committed|observables|msg0|sup0 | 0.184 | -1.839 | 0.01 |
| committed|observables|msg10|sup0 | 0.177 | -1.905 | 0.01 |
| committed|observables|msg3|sup0 | 0.179 | -1.936 | 0.01 |
| committed|observables|msg1|sup0 | 0.192 | -2.011 | 0.02 |
| neutral|rule|msg0|sup0 | 0.605 | -0.515 | 0.12 |
| neutral|observables|msg0|sup0 | 0.605 | -0.515 | 0.12 |
| neutral|observables|msg10|sup0 | 0.588 | -0.548 | 0.12 |
| neutral|observables|msg3|sup0 | 0.581 | -0.571 | 0.16 |
| neutral|observables|msg1|sup0 | 0.574 | -0.623 | 0.25 |
| mix|rule|msg0|sup0 | 0.379 | -1.092 | 0.10 |
| mix|observables|msg0|sup0 | 0.382 | -1.015 | 0.02 |
| mix|observables|msg10|sup0 | 0.388 | -1.012 | 0.03 |
| mix|observables|msg3|sup0 | 0.378 | -1.056 | 0.03 |
| mix|observables|msg1|sup0 | 0.374 | -1.135 | 0.04 |
| mix|observables|msg10|sup10 | 0.388 | -1.012 | 0.03 |
| mix|observables|msg3|sup3 | 0.384 | -1.038 | 0.03 |
| mix|observables|msg1|sup1 | 0.376 | -1.129 | 0.04 |

## Part 3: neutral-drift control (world closed for everyone, symmetric start, 40 seeds)

| arm | mean abs(P(M>0)-.5) at end | share of seeds ending > .5 | median t to abs dev .25 | share of seeds reaching .25 | within-seed agent SD | across-seed SD |
|---|---|---|---|---|---|---|
| N24|neutral|m0 | 0.000 | 1.00 | 600 | 0.00 | 0.000 | 0.000 |
| N24|neutral|m1 | 0.124 | 0.53 | 18 | 1.00 | 0.000 | 0.158 |
| N24|neutral|m10 | 0.036 | 0.35 | 600 | 0.00 | 0.000 | 0.044 |
| N24|neutral|m3 | 0.081 | 0.42 | 250 | 0.85 | 0.000 | 0.097 |
| N24|pluralist|m0 | 0.004 | 1.00 | 600 | 0.00 | 0.000 | 0.000 |
| N24|pluralist|m1 | 0.119 | 0.55 | 29 | 1.00 | 0.000 | 0.151 |
| N24|pluralist|m10 | 0.037 | 0.38 | 600 | 0.00 | 0.000 | 0.046 |
| N24|pluralist|m3 | 0.076 | 0.33 | 376 | 0.70 | 0.000 | 0.092 |
| N50|neutral|m0 | 0.000 | 1.00 | 600 | 0.00 | 0.000 | 0.000 |
| N50|neutral|m1 | 0.094 | 0.42 | 68 | 0.97 | 0.000 | 0.112 |
| N50|neutral|m10 | 0.026 | 0.45 | 600 | 0.00 | 0.000 | 0.031 |
| N50|neutral|m3 | 0.054 | 0.35 | 600 | 0.00 | 0.000 | 0.066 |
| N50|pluralist|m0 | 0.004 | 1.00 | 600 | 0.00 | 0.000 | 0.000 |
| N50|pluralist|m1 | 0.090 | 0.40 | 129 | 0.95 | 0.000 | 0.106 |
| N50|pluralist|m10 | 0.026 | 0.50 | 600 | 0.00 | 0.000 | 0.031 |
| N50|pluralist|m3 | 0.053 | 0.38 | 600 | 0.00 | 0.000 | 0.065 |
| N8|neutral|m0 | 0.000 | 1.00 | 600 | 0.00 | 0.000 | 0.000 |
| N8|neutral|m1 | 0.204 | 0.60 | 6 | 1.00 | 0.000 | 0.242 |
| N8|neutral|m10 | 0.077 | 0.40 | 259 | 0.72 | 0.000 | 0.092 |
| N8|neutral|m3 | 0.124 | 0.53 | 18 | 1.00 | 0.000 | 0.158 |
| N8|pluralist|m0 | 0.004 | 1.00 | 600 | 0.00 | 0.000 | 0.000 |
| N8|pluralist|m1 | 0.203 | 0.60 | 6 | 1.00 | 0.000 | 0.240 |
| N8|pluralist|m10 | 0.071 | 0.38 | 459 | 0.57 | 0.000 | 0.086 |
| N8|pluralist|m3 | 0.118 | 0.55 | 29 | 1.00 | 0.000 | 0.151 |
