# PromiseGuard: Proactive Intervention Policy Simulation Report

**Evaluation Test Dataset**: Chronological Held-out Test Set ($N = 6,838$ orders)  
**Total Ground-Truth Delayed Orders**: 440 orders (6.43%)  

---

## 1. Operational Policy Comparison Matrix

| Operational Policy | Alert Rate | True Delays Captured | Recall % | False Alerts | Precision % | Missed Delays | Operational Impact |
|---|---|---|---|---|---|---|---|
| **Policy A: Fixed Promise (No Proactive Action)** | 0.0% (0) | **0** | **0.0%** | 0 | 0.0% | 440 | Zero dispatch overhead, maximum late customer dissatisfaction |
| **Policy B: Risk-Threshold Alerting (T=0.57)** | 10.78% (737) | **399** | **90.68%** | 338 | 54.14% | 41 | Moderate automated customer notification volume (737 alerts) |
| **Policy C: Interval-Aware ETA (Width >= 13.0 min)** | 45.04% (3,080) | **129** | **29.32%** | 2,951 | 4.19% | 311 | Zero dispatch overhead; sets honest customer expectations upfront |
| **Policy D: Operations Dispatcher Review (Top 10% Risk Queue)** | 10.0% (684) | **388** | **88.18%** | 296 | 56.73% | 52 | Requires dispatch personnel capacity for 684 high-priority orders |

---

## 2. Threshold Sensitivity Analysis (Trade-off: False Alerts vs Missed Delays)

| Risk Threshold ($T$) | Alert Count | Recall (Delays Caught) | Precision | False Alerts | Missed Delays | Illustrative Cost Index |
|---|---|---|---|---|---|---|
| $T = 0.1$ | 1,395 (20.4%) | **99.32%** | 31.33% | 958 | 3 | 973.0 |
| $T = 0.15$ | 1,218 (17.81%) | **99.32%** | 35.88% | 781 | 3 | 796.0 |
| $T = 0.2$ | 1,106 (16.17%) | **99.09%** | 39.42% | 670 | 4 | 690.0 |
| $T = 0.25$ | 1,027 (15.02%) | **98.41%** | 42.16% | 594 | 7 | 629.0 |
| $T = 0.3$ | 972 (14.21%) | **98.18%** | 44.44% | 540 | 8 | 580.0 |
| $T = 0.35$ | 926 (13.54%) | **97.27%** | 46.22% | 498 | 12 | 558.0 |
| $T = 0.4$ | 877 (12.83%) | **95.91%** | 48.12% | 455 | 18 | 545.0 |
| $T = 0.45$ | 847 (12.39%) | **94.55%** | 49.11% | 431 | 24 | 551.0 |
| $T = 0.5$ | 804 (11.76%) | **93.86%** | 51.37% | 391 | 27 | 526.0 |
| $T = 0.55$ | 754 (11.03%) | **91.59%** | 53.45% | 351 | 37 | 536.0 |
| $T = 0.6$ | 714 (10.44%) | **90.0%** | 55.46% | 318 | 44 | 538.0 |
| $T = 0.65$ | 674 (9.86%) | **87.95%** | 57.42% | 287 | 53 | 552.0 |
| $T = 0.7$ | 630 (9.21%) | **85.68%** | 59.84% | 253 | 63 | 568.0 |
| $T = 0.75$ | 582 (8.51%) | **82.95%** | 62.71% | 217 | 75 | 592.0 |
| $T = 0.8$ | 532 (7.78%) | **80.68%** | 66.73% | 177 | 85 | 602.0 |
| $T = 0.85$ | 479 (7.0%) | **77.73%** | 71.4% | 137 | 98 | 627.0 |
| $T = 0.9$ | 404 (5.91%) | **69.55%** | 75.74% | 98 | 134 | 768.0 |

---

## 3. Managerial Findings & Academic Reality Check
1. **The Cost of Complacency (Policy A)**: Doing nothing results in 100% of delayed orders reaching customers as unpleasant surprises, maximizing customer support contacts and refund friction.
2. **Targeted Risk Alerting (Policy B)**: By triggering updates when predicted probability exceeds $T^* = 0.57$, the platform intercepts over 90% of genuine delays, giving customers realistic updates before the promise boundary passes.
3. **Transparent Range Communication (Policy C)**: Showing prediction intervals during high uncertainty eliminates the false expectation of a deterministic minute, aligning customer expectations with operational realities without generating alert fatigue.
4. **Offline Simulation Disclaimer**: This simulation evaluates policy trade-offs under historical operational conditions. It does NOT claim causal proof that sending an update increases customer retention or reduces refund requests; live validation requires a randomized controlled trial (A/B testing).
