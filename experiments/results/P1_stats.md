# P1 statistics: paired-by-seed contrasts (P1b-P1e)

Seeds 0-19 (paired: identical observation noise across arms). Outcome: community-mean P(M>0) at round 599, averaged per seed over the arm's gate cells (and seer sets in P1d/P1e). Effect = mean paired difference A - B with 95% bootstrap CI (10k resamples over seeds); d_z = mean/sd of paired differences; p = Wilcoxon signed-rank; Holm across all rows below. 'cells +/-' = gate cells (x seer sets) individually significant after Holm within that contrast.

| study | contrast (A - B) | mean A | mean B | diff [95% CI] | d_z | p | p_Holm | Holm | cells +/- |
|---|---|---|---|---|---|---|---|---|---|
| P1b | pluralist - committed monist | 0.512 | 0.175 | +0.338 [+0.331, +0.344] | 21.75 | 1.9e-06 | 9.3e-05 | yes | 15/10 of 25 |
| P1b | pluralist - neutral monist | 0.512 | 0.617 | -0.105 [-0.110, -0.099] | -8.42 | 1.9e-06 | 9.3e-05 | yes | 13/12 of 25 |
| P1b | pluralist(forget .95) - committed monist | 0.613 | 0.175 | +0.438 [+0.433, +0.444] | 36.06 | 1.9e-06 | 9.3e-05 | yes | 25/0 of 25 |
| P1c | monists 50% committed - 0% committed | 0.341 | 0.617 | -0.276 [-0.279, -0.272] | -35.06 | 1.9e-06 | 9.3e-05 | yes | 0/25 of 25 |
| P1c | 25% pluralists - 0% (monists 50:50) | 0.449 | 0.341 | +0.108 [+0.099, +0.117] | 5.03 | 1.9e-06 | 9.3e-05 | yes | 25/0 of 25 |
| P1c | 75% pluralists - 25% (monists 50:50) | 0.583 | 0.449 | +0.134 [+0.123, +0.144] | 5.53 | 1.9e-06 | 9.3e-05 | yes | 18/7 of 25 |
| P1c | 75% pluralists (50:50) - 100% pluralists | 0.583 | 0.512 | +0.071 [+0.066, +0.076] | 5.83 | 1.9e-06 | 9.3e-05 | yes | 12/13 of 25 |
| P1c | 25% pluralists at observers - none at observers | 0.502 | 0.418 | +0.084 [+0.078, +0.090] | 5.88 | 1.9e-06 | 9.3e-05 | yes | 23/1 of 25 |
| P1c | drag: pluralists in 25%-pluralist/committed community - pure pluralists | 0.331 | 0.512 | -0.181 [-0.189, -0.173] | -9.18 | 1.9e-06 | 9.3e-05 | yes | 12/13 of 25 |
| P1c | pull: committed in 75%-pluralist community - pure committed | 0.505 | 0.175 | +0.331 [+0.325, +0.337] | 23.80 | 1.9e-06 | 9.3e-05 | yes | 25/0 of 25 |
| P1d | relative - fixed W [all_pluralist, mode b, D=10] | 0.642 | 0.558 | +0.084 [+0.081, +0.089] | 9.46 | 1.9e-06 | 9.3e-05 | yes | 17/1 of 27 |
| P1d | agree - fixed W [all_pluralist, mode b, D=10] | 0.421 | 0.558 | -0.137 [-0.140, -0.132] | -14.70 | 1.9e-06 | 9.3e-05 | yes | 7/9 of 27 |
| P1d | relative - fixed W [all_pluralist, mode a, D=10] | 0.619 | 0.524 | +0.096 [+0.087, +0.107] | 4.22 | 1.9e-06 | 9.3e-05 | yes | 16/1 of 27 |
| P1d | D=1 - D=50 [all_pluralist, relative, mode b] | 0.644 | 0.625 | +0.019 [+0.015, +0.022] | 2.28 | 1.9e-06 | 9.3e-05 | yes | 11/3 of 27 |
| P1e | full - attention-weighted reveals [all_pluralist, fixed W, mode b, D=10] | 0.986 | 0.558 | +0.428 [+0.423, +0.434] | 36.72 | 1.9e-06 | 9.3e-05 | yes | 27/0 of 27 |
| P1e | relative(k=30) - fixed W [all_pluralist, full, D=10] | 0.988 | 0.986 | +0.002 [-0.000, +0.005] | 0.38 | 0.25 | 1 | no | 1/2 of 27 |
| P1e | D=1 - D=50 [all_pluralist, relative k=30, full] | 0.986 | 0.987 | -0.001 [-0.007, +0.005] | -0.07 | 0.5 | 1 | no | 0/0 of 27 |
| P1e | D=1 - D=50 [all_pluralist, fixed W, full] | 0.987 | 0.986 | +0.001 [-0.004, +0.007] | 0.08 | 0.84 | 1 | no | 0/0 of 27 |
| P1e | relative(k=30) - fixed W [all_pluralist, mode a, D=10] | 0.619 | 0.524 | +0.095 [+0.072, +0.117] | 1.77 | 9.5e-06 | 0.00017 | yes | 14/0 of 27 |
| P1e | sparse reveals: relative(k=30) - fixed W [all_pluralist, every 10th, full, D=1] | 0.935 | 0.920 | +0.015 [+0.012, +0.018] | 2.29 | 1.9e-06 | 9.3e-05 | yes | 13/0 of 27 |
| P1e | sparse reveals: D=1 - D=50 [all_pluralist, relative k=30, every 10th] | 0.935 | 0.929 | +0.006 [-0.009, +0.019] | 0.17 | 0.55 | 1 | no | 0/0 of 27 |
| P1e | sparse reveals: D=1 - D=50 [all_pluralist, fixed W, every 10th] | 0.920 | 0.909 | +0.011 [-0.004, +0.025] | 0.32 | 0.22 | 1 | no | 3/0 of 27 |
| P1e | agree - fixed W [all_pluralist, full, D=10] | 0.828 | 0.986 | -0.158 [-0.173, -0.145] | -4.72 | 1.9e-06 | 9.3e-05 | yes | 1/6 of 27 |
| P1d | relative - fixed W [half_pluralist_committed, mode b, D=10] | 0.457 | 0.424 | +0.033 [+0.031, +0.035] | 6.44 | 1.9e-06 | 9.3e-05 | yes | 26/0 of 27 |
| P1d | agree - fixed W [half_pluralist_committed, mode b, D=10] | 0.355 | 0.424 | -0.069 [-0.075, -0.062] | -4.44 | 1.9e-06 | 9.3e-05 | yes | 0/27 of 27 |
| P1d | relative - fixed W [half_pluralist_committed, mode a, D=10] | 0.413 | 0.379 | +0.033 [+0.030, +0.036] | 4.68 | 1.9e-06 | 9.3e-05 | yes | 24/0 of 27 |
| P1d | D=1 - D=50 [half_pluralist_committed, relative, mode b] | 0.459 | 0.450 | +0.009 [+0.003, +0.015] | 0.61 | 0.0083 | 0.13 | no | 5/2 of 27 |
| P1e | full - attention-weighted reveals [half_pluralist_committed, fixed W, mode b, D=10] | 0.944 | 0.424 | +0.520 [+0.511, +0.529] | 24.25 | 1.9e-06 | 9.3e-05 | yes | 27/0 of 27 |
| P1e | relative(k=30) - fixed W [half_pluralist_committed, full, D=10] | 0.911 | 0.944 | -0.033 [-0.040, -0.027] | -2.23 | 1.9e-06 | 9.3e-05 | yes | 1/15 of 27 |
| P1e | D=1 - D=50 [half_pluralist_committed, relative k=30, full] | 0.920 | 0.920 | -0.000 [-0.020, +0.021] | -0.00 | 0.87 | 1 | no | 0/0 of 27 |
| P1e | D=1 - D=50 [half_pluralist_committed, fixed W, full] | 0.946 | 0.943 | +0.003 [-0.013, +0.021] | 0.08 | 0.73 | 1 | no | 0/0 of 27 |
| P1e | relative(k=30) - fixed W [half_pluralist_committed, mode a, D=10] | 0.621 | 0.379 | +0.241 [+0.218, +0.265] | 4.28 | 1.9e-06 | 9.3e-05 | yes | 23/0 of 27 |
| P1e | sparse reveals: relative(k=30) - fixed W [half_pluralist_committed, every 10th, full, D=1] | 0.621 | 0.657 | -0.036 [-0.100, +0.034] | -0.23 | 0.39 | 1 | no | 2/0 of 27 |
| P1e | sparse reveals: D=1 - D=50 [half_pluralist_committed, relative k=30, every 10th] | 0.621 | 0.604 | +0.017 [-0.015, +0.050] | 0.23 | 0.37 | 1 | no | 0/0 of 27 |
| P1e | sparse reveals: D=1 - D=50 [half_pluralist_committed, fixed W, every 10th] | 0.657 | 0.634 | +0.023 [-0.005, +0.049] | 0.36 | 0.12 | 1 | no | 0/0 of 27 |
| P1e | agree - fixed W [half_pluralist_committed, full, D=10] | 0.831 | 0.944 | -0.113 [-0.139, -0.088] | -1.91 | 1.9e-06 | 9.3e-05 | yes | 0/18 of 27 |
| P1d | relative - fixed W [all_committed, mode b, D=10] | 0.211 | 0.211 | +0.000 [+0.000, +0.000] | 2.25 | 3.8e-06 | 9.3e-05 | yes (negligible) | 22/5 of 27 |
| P1d | agree - fixed W [all_committed, mode b, D=10] | 0.160 | 0.211 | -0.052 [-0.054, -0.049] | -9.85 | 1.9e-06 | 9.3e-05 | yes | 0/27 of 27 |
| P1d | relative - fixed W [all_committed, mode a, D=10] | 0.184 | 0.184 | +0.000 [+0.000, +0.000] | 2.12 | 5.7e-06 | 0.00011 | yes (negligible) | 24/3 of 27 |
| P1d | D=1 - D=50 [all_committed, relative, mode b] | 0.211 | 0.211 | +0.000 [-0.002, +0.002] | 0.01 | 0.96 | 1 | no | 0/0 of 27 |
| P1e | full - attention-weighted reveals [all_committed, fixed W, mode b, D=10] | 0.857 | 0.211 | +0.646 [+0.622, +0.668] | 11.88 | 1.9e-06 | 9.3e-05 | yes | 27/0 of 27 |
| P1e | relative(k=30) - fixed W [all_committed, full, D=10] | 0.857 | 0.857 | -0.000 [-0.000, -0.000] | -1.99 | 1.9e-06 | 9.3e-05 | yes (negligible) | 0/27 of 27 |
| P1e | D=1 - D=50 [all_committed, relative k=30, full] | 0.861 | 0.857 | +0.004 [-0.026, +0.036] | 0.06 | 0.93 | 1 | no | 0/0 of 27 |
| P1e | D=1 - D=50 [all_committed, fixed W, full] | 0.861 | 0.857 | +0.004 [-0.026, +0.036] | 0.06 | 0.93 | 1 | no | 0/0 of 27 |
| P1e | relative(k=30) - fixed W [all_committed, mode a, D=10] | 0.184 | 0.184 | +0.000 [+0.000, +0.000] | 2.12 | 5.7e-06 | 0.00011 | yes (negligible) | 24/3 of 27 |
| P1e | sparse reveals: relative(k=30) - fixed W [all_committed, every 10th, full, D=1] | 0.321 | 0.321 | +0.000 [+0.000, +0.000] | 0.80 | 0.0037 | 0.062 | no | 13/6 of 27 |
| P1e | sparse reveals: D=1 - D=50 [all_committed, relative k=30, every 10th] | 0.321 | 0.298 | +0.024 [-0.009, +0.055] | 0.31 | 0.22 | 1 | no | 0/0 of 27 |
| P1e | sparse reveals: D=1 - D=50 [all_committed, fixed W, every 10th] | 0.321 | 0.298 | +0.024 [-0.009, +0.055] | 0.31 | 0.22 | 1 | no | 0/0 of 27 |
| P1e | agree - fixed W [all_committed, full, D=10] | 0.756 | 0.857 | -0.102 [-0.122, -0.085] | -2.27 | 1.9e-06 | 9.3e-05 | yes | 0/10 of 27 |

Notes: with 20 seeds the smallest attainable two-sided Wilcoxon p is ~1.9e-6, so the Holm threshold for the most significant of the contrasts above is 0.05/49. Contrasts with all-zero paired differences get p = 1. 'yes (negligible)' marks Holm-significant contrasts whose mean paired difference is below 0.005 in P(M>0) - numerically consistent but practically null (e.g. trust rules in an all-committed community, where nobody forecasts correctly, differ from fixed W only at the 1e-4 level). Differences are tiny d_z-inflated because paired seeds share observation noise; read the CI width, not p alone.
