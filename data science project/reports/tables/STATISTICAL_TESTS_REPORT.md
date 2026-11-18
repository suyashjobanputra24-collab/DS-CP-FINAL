# PromiseGuard: Consolidated Statistical Hypothesis Testing Report

**Course**: MD3135 Data Science | Third Year Computer Engineering  
**Academic Significance Level**: $\alpha = 0.05$  
**Report Type**: Formal Inferential Statistics and Hypothesis Evaluation  

---

## Executive Summary of Hypothesis Results

| Hypothesis ID | Research Question | Test Method | Test Statistic | p-value | Effect Size | Academic Conclusion |
|---|---|---|---|---|---|---|
| **H1** | Traffic density vs Delivery Duration | Kruskal-Wallis & Welch's t | $H = 8261.549$ | $< 0.001$ | $\eta^2 = 0.1828$ | **Reject $H_0$**: Significant impact |
| **H2** | Weather conditions vs Delivery Duration | Kruskal-Wallis H | $H = 2784.72$ | $< 0.001$ | Multi-group | **Reject $H_0$**: Significant impact |
| **H3** | Peak hours vs Off-peak hours | Welch's t & Mann-Whitney | $t = 57.581$ | $< 0.001$ | $d = 0.542$ | **Reject $H_0$**: Statistically significant |
| **H4** | Vehicle type vs Delivery Duration | Kruskal-Wallis H | $H = 1080.526$ | $< 0.05$ | Multi-group | **Reject $H_0$**: Significant difference |
| **H5** | Delivery time vs Customer Rating | Spearman Rank & Mann-Whitney | $\rho = 0.0007$ | $0.9809$ | $\chi^2 = 0.06$ | **Fail to Reject H0 (No Significant Linear Association)** |

---

## In-Depth Analysis of Individual Hypotheses

### 1. Hypothesis H1: Road Traffic Density Impact
- **Null Hypothesis ($H_0$)**: $\mu_{\text{Low}} = \mu_{\text{Medium}} = \mu_{\text{High}} = \mu_{\text{Jam}}$
- **Alternative Hypothesis ($H_1$)**: At least two traffic conditions have different mean delivery durations.
- **Group Details**:
  - **Low Traffic**: Mean = 21.27 min, Median = 20.0 min, Std = 6.8 min (N = 15,476)
  - **Medium Traffic**: Mean = 26.69 min, Median = 26.0 min, Std = 8.61 min (N = 11,546)
  - **High Traffic**: Mean = 27.24 min, Median = 27.0 min, Std = 8.4 min (N = 4,423)
  - **Jam Traffic**: Mean = 31.18 min, Median = 31.0 min, Std = 9.94 min (N = 14,139)

- **Statistical Findings**: Delivery duration exhibits a statistically significant difference across traffic conditions (Kruskal-Wallis H=8261.55, p < 0.001; eta-squared=0.183). Jam traffic conditions add an average of 9.9 minutes compared to Low traffic, representing a large effect size (Cohen's d=1.16).

---

### 2. Hypothesis H2: Weather Conditions
- **Null Hypothesis ($H_0$)**: Delivery duration distributions are identical across weather conditions.
- **Statistical Findings**: Weather conditions have a statistically significant relationship with delivery duration (Kruskal-Wallis H=2784.72, p < 0.001). Adverse weather such as Fog, Stormy, and Sandstorms consistently increases delivery duration compared to Sunny conditions (by ~6.7 minutes).

---

### 3. Hypothesis H3: Peak-Hour vs Off-Peak Operational Load
- **Null Hypothesis ($H_0$)**: $\mu_{\text{Peak}} = \mu_{\text{Off-Peak}}$
- **Observed Peak Mean**: 28.87 min vs Off-Peak Mean: 23.94 min
- **95% Confidence Interval for Difference**: [4.76, 5.1] minutes.
- **Statistical Findings**: Peak hours (Lunch and Dinner) show a statistically significant increase in delivery time (mean difference = 4.93 mins, 95% CI [4.76, 5.10], p < 0.001). While statistically significant due to large sample size (N=45,584), the practical effect size is modest (Cohen's d=0.54), confirming that traffic and distance remain the dominant drivers of total variance.

---

### 4. Hypothesis H5: Customer Rating & Satisfaction (FoodHub)
- **Null Hypothesis ($H_0$)**: Customer rating is monotonically independent of delivery duration.
- **Observations**: 1,162 rated orders out of 1,898 total orders (38.78% unrated).
- **Spearman Rank Correlation**: $\rho = 0.0007$ ($p = 0.9809$)
- **Statistical Findings**: In the FoodHub dataset (N=1162 rated orders), the rank correlation between delivery time and customer rating is weak (Spearman rho = 0.001, p = 0.981). Dissatisfied customers (rating <= 3) had an average delivery time of 24.6 mins compared to 24.1 mins for satisfied customers (rating >= 4). Importantly, 38.78% of orders had 'Not given' ratings. This empirical finding demonstrates that satisfaction in instant food delivery is multi-factorial (food quality, order accuracy, packaging temperature) and cannot be accurately predicted solely from transit duration.

---

## Methodological Safeguards & Academic Disclaimers
1. **Statistical Significance vs Practical Relevance**: Due to large sample sizes in operational delivery logs ($N > 45,000$), minute differences can yield $p < 0.001$ without representing substantial operational impact. Effect sizes (Cohen's $d$, $\eta^2$) are reported alongside $p$-values to prevent misleading claims.
2. **Correlation vs Causation**: Observational correlations between weather, traffic, and duration do not account for latent confounders (e.g., driver route unfamiliarity, restaurant kitchen surge).
