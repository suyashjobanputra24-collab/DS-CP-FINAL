# PromiseGuard: Comprehensive Academic Viva Defense & Examination Guide

**Course**: MD3135 Data Science | Third Year Computer Engineering  
**Project ID**: MDM-DS-039  
**Students**: Suyash Jobanputra & Shreyash Jobanputra  
**Guide**: Dr. Kapil N. Vhatkar  
**Academic Year**: 2026–27  

---

## 1. Project Synopsis in Simple Plain English
*(How to explain the project to the examiner in the first 60 seconds)*

> "Respected examiners, our project is called **PromiseGuard**. 
> In instant food delivery platforms like Zomato, Swiggy, and Uber Eats, an Estimated Time of Arrival (ETA) is shown to customers when they order. 
> However, existing systems typically present a single fixed number—such as '35 minutes'—which hides operational uncertainty. 
> Two orders might both be promised in 35 minutes; one arrives smoothly in 36 minutes, but another gets delayed by 20 minutes due to unexpected kitchen backlogs or rain-induced traffic jams. 
>
> In PromiseGuard, we built an explainable data-science framework that:
> 1. Accurately predicts delivery duration using machine learning.
> 2. Quantifies uncertainty by producing a **calibrated 90% prediction interval** (e.g. 32–44 minutes) using Quantile Regression and Conformal Prediction.
> 3. Identifies **Promise-Break Risk**—predicting the probability that an order will miss its stated delivery commitment.
> 4. Explains individual predictions using **SHAP (Shapley Additive exPlanations)**.
> 5. Evaluates proactive customer updates and operational response policies through an offline decision simulator.
> 6. Studies genuine customer ratings to test whether delivery duration directly determines customer satisfaction."

---

## 2. Core Concepts Explained From Absolute Zero

### A. What is ETA Prediction and Why is it Hard?
ETA stands for **Estimated Time of Arrival**. It is the total elapsed time between when a customer places an order and when the food is physically handed over at their doorstep.
It is difficult because it is composed of multiple uncertain stages:
1. **Restaurant Preparation Time (KPT)**: Kitchen workload, order complexity, chef capacity.
2. **Dispatch & Assignment Delay**: Finding an available delivery partner nearby.
3. **First-Mile Travel**: Partner traveling to the restaurant.
4. **Pickup Waiting Time**: Partner waiting at the restaurant counter for food packing.
5. **Last-Mile Transit**: Driving through traffic lights, road closures, and weather.
6. **Handover Time**: Parking, elevator access, security gate check, customer door buzzer.

### B. What is the Difference Between ETA Error and Promise Error?
- **ETA Prediction Error**: $\text{Actual Duration} - \text{Predicted Duration}$ ($y - \hat{y}$). It measures the raw statistical error of the regression model.
- **Promise Error**: $\text{Actual Arrival} - \text{Promised SLA Boundary}$ ($y - \text{SLA Promise}$). It measures the customer-facing service breach. An order can have an ETA error of 2 minutes, but if it crosses the customer promise boundary, it is perceived as a broken promise.

### C. What is a Prediction Interval?
A **point prediction** outputs a single number: "Your order will arrive in 35 minutes."  
A **prediction interval** outputs a range: "Your order will arrive between 31 and 42 minutes with 90% statistical confidence."  
It captures how much variance is expected for that specific order under current weather, traffic, and distance conditions.

### D. What is Conformal Prediction?
Conformal Prediction is a modern statistical framework that takes any machine learning model and constructs prediction intervals with **mathematically guaranteed finite-sample coverage**, without making unrealistic assumptions that errors are normally distributed (Gaussian).  
In our project, on 6,838 test orders, our 90% conformal interval achieved **90.33% empirical coverage**, exactly meeting its theoretical guarantee.

### E. What does SHAP (Shapley Additive exPlanations) do?
SHAP originates from cooperative game theory (Lloyd Shapley, Nobel Laureate). It treats each input feature (distance, weather, traffic) as a "player" in a game whose payout is the final predicted delivery time. SHAP measures how much each feature contributed to shifting the prediction away from the global dataset average.  
*Key Viva Note*: SHAP explains **model behavior**, not physical causation!

---

## 3. Top 25 Examiner Questions & Model Answers

### Category 1: Problem Definition & Novelty
#### Q1: "Food delivery companies like Zomato and Swiggy already have ETA prediction. Did you reinvent it?"
**Answer**:
"No, sir. We explicitly state in our report that major platforms already possess proprietary multi-stage ETA systems, dynamic buffers, and dispatch engines. Our academic contribution is NOT claiming to replace proprietary industrial engines. Instead, our research contribution is an **open, transparent, and reproducible framework** that studies prediction reliability across operational segments, quantifies uncertainty with conformal prediction, explains risk drivers with SHAP, evaluates genuine customer satisfaction coupling, and simulates proactive intervention policies on accessible data."

#### Q2: "What is your primary regression target?"
**Answer**:
"`Time_taken (min)`—the total elapsed delivery duration in minutes from order placement to completion. The dataset spans 45,584 orders with a mean of 26.29 minutes and standard deviation of 9.38 minutes."

#### Q3: "What data did you use? Did you fabricate any data?"
**Answer**:
"We strictly adhered to non-negotiable research integrity rules and used two real-world datasets:
1. An operational delivery logistics benchmark with 45,584 orders tracking delivery partner demographics, GPS coordinates, weather conditions, road traffic density, vehicle types, and delivery durations.
2. The FoodHub order experience dataset with 1,898 orders tracking food preparation time, delivery duration, and genuine post-delivery customer ratings (1 to 5 stars).
We never fabricated ratings, never relabeled delivery-partner ratings as customer ratings, and preserved all unrated orders as an explicit category."

---

### Category 2: Machine Learning & Preprocessing
#### Q4: "How did you prevent data leakage during train/test splitting?"
**Answer**:
"Rather than using an arbitrary random shuffle split, we implemented a **strict chronological (temporal) split** sorted by `Order_Date` and order timestamp:
- **Training Set**: Earliest 70% of orders ($N = 31,908$).
- **Validation Set**: Middle 15% of orders ($N = 6,838$).
- **Test Set**: Latest 15% of orders ($N = 6,838$).
This simulates real-world production deployment: training on past history and predicting forward into the future without looking ahead. Furthermore, all imputers, scalers, and encoders were fit strictly on the training set."

#### Q5: "How did you calculate delivery distance from coordinates?"
**Answer**:
"We used the **Haversine formula**, which computes the great-circle spherical distance between latitude/longitude pairs on Earth in kilometers. We also audited the raw coordinates: certain GPS sensor dropouts had coordinates near zero or inverted negative signs in latitude; our preprocessing pipeline corrected coordinate signs and imputed missing sensor records using training medians."

#### Q6: "Why did you compare multiple models instead of just using Random Forest directly?"
**Answer**:
"In rigorous Data Science, advanced algorithms must justify their complexity against simple baselines:
1. **Global Median Baseline**: Predicts 26.0 min for all orders $\rightarrow$ Test MAE = 7.48 min, $R^2 \approx 0$.
2. **Segment Median Baseline**: Predicts median by traffic and distance tier $\rightarrow$ Test MAE = 6.48 min, $R^2 = 0.235$.
3. **Ridge Linear Regression**: Adds linear features $\rightarrow$ Test MAE = 4.85 min, $R^2 = 0.570$.
4. **Random Forest Regressor**: Captures non-linearities $\rightarrow$ Test MAE = 3.25 min, $R^2 = 0.808$.
5. **HistGradientBoostingRegressor**: State-of-the-art tree ensemble $\rightarrow$ Test MAE = 3.26 min, $R^2 = 0.808$.
This proves that tree-based gradient boosting achieves a **56% error reduction** over baseline rules."

#### Q7: "What is the difference between MAE and RMSE?"
**Answer**:
"MAE (Mean Absolute Error) is the linear average of absolute errors in minutes; it is robust to outliers and easily understood by stakeholders.  
RMSE (Root Mean Squared Error) squares errors before taking the square root; it heavily penalizes large errors. In our test set, HistGradientBoosting achieved an MAE of 3.26 minutes and an RMSE of 4.08 minutes."

---

### Category 3: Reliability & Uncertainty Quantification
#### Q8: "What did your Prediction Reliability analysis discover?"
**Answer**:
"We analyzed directional error and segment vulnerability:
- **Directional Balance**: 36.0% of orders were accurate within $\pm 2$ minutes, 30.4% were underpredicted (late delivery), and 33.7% were overpredicted (early arrival).
- **Segment Vulnerability**: In Low traffic, MAE is 2.77 min and severe late errors ($\ge 10$ min late) happen in only 0.21% of orders. But in **Jam traffic**, MAE rises to 3.60 min and severe late errors jump eight-fold to 1.70%.
- Adverse weather (Stormy, Sandstorms, Fog) similarly increases variance and tail risk."

#### Q9: "How does Quantile Regression work?"
**Answer**:
"Standard regression minimizes Mean Squared Error or Mean Absolute Error to estimate the conditional mean or median. Quantile Regression minimizes the **asymmetric pinball loss** (quantile loss) to estimate specific percentiles of the output distribution. We trained two models: one at $\alpha = 0.05$ (lower 5th percentile) and one at $\alpha = 0.95$ (upper 95th percentile). The range between them forms a 90% prediction interval whose width adapts dynamically based on traffic and distance."

#### Q10: "How did your prediction intervals perform on test data?"
**Answer**:
"On our chronological held-out test set ($N = 6,838$):
- **Quantile Gradient Boosting**: Achieved **87.34% empirical coverage** with an adaptive average width of 12.74 minutes.
- **Split-Conformal Prediction**: Achieved **90.33% empirical coverage** with a calibrated threshold of $\pm 6.52$ minutes (width = 13.04 min).
Both methods proved highly reliable, confirming our hypothesis that prediction uncertainty can be statistically bounded."

---

### Category 4: Promise Risk & Customer Satisfaction
#### Q11: "How did you define the Promise-Break target?"
**Answer**:
"We defined an operational SLA promise boundary based on industry baseline rules:  
$\text{Promised SLA} = 15.0\text{ min (base prep/handover)} + (2.2 \times \text{distance}) + \text{traffic buffer}$ (2 min for Low, 6 for Medium, 12 for High, 18 for Jam).  
An order is labeled $\text{Late} = 1$ if actual delivery exceeded this SLA boundary, and $0$ otherwise. Ground-truth late orders comprised 6.87% of all orders."

#### Q12: "Why can't you use standard 50% threshold for classification?"
**Answer**:
"Because the class is imbalanced (only ~6.9% late orders), predicting all orders as 'on time' would give 93.1% accuracy but catch 0 late orders!  
We used balanced class weights and evaluated the cost trade-off on our validation set. Since missing a late order is much more costly to customer trust than sending a cautious update, we optimized an operational cost curve ($5 \times \text{FN} + 1 \times \text{FP}$), finding an optimal decision threshold of $T^* = 0.57$. At this threshold, our model catches **90.7% of all delayed orders** (Recall) with 54.1% Precision and an ROC-AUC of 0.9829."

#### Q13: "What did you discover about Customer Satisfaction in FoodHub?"
**Answer**:
"This was one of our most important scientific findings:  
1. **Unrated Orders**: 38.78% of customers left 'Not given' ratings. We did not artificially impute them as dissatisfied.
2. **Weak Coupling with Duration**: The average delivery duration for 5-star satisfied ratings was 24.16 minutes, compared to 24.64 minutes for 3-star ratings—a difference of less than 30 seconds!
3. Classifiers trained to predict customer satisfaction solely from delivery duration achieved an ROC-AUC of ~0.46 (no better than random guessing).  
This proves that **delivery time alone does not determine customer satisfaction**; food temperature, order correctness, packaging quality, and food taste dominate customer perception."

---

### Category 5: Explainability & Simulation
#### Q14: "What were the most important features in SHAP and Permutation tests?"
**Answer**:
"Across both Permutation Importance and SHAP values:
1. **Road Traffic Density** ('Jam' and 'High'): Strongest driver, shifting ETAs upwards by +4 to +8 minutes.
2. **Weather Conditions**: Stormy, Fog, and Sandstorms contributed +2 to +4 minutes.
3. **Haversine Distance**: Adds approximately 1.4 minutes per km.
4. **Courier Historical Rating & Age**: Experienced, high-rated couriers delivered faster.
5. **Multiple Deliveries**: Concurrently carrying 2 or 3 orders added +3 to +6 minutes of delivery latency."

#### Q15: "What is the Proactive Intervention Simulator?"
**Answer**:
"It is an offline decision-support simulator that evaluated 4 operational policies on the 6,838 test orders:
- **Policy A (Fixed Promise)**: Passive baseline. Captures 0 delays early; all 440 delays arrive late without prior warning.
- **Policy B (Risk Threshold at $T^* = 0.57$)**: Automatically alerts customers with a revised ETA when delay risk is high. Intercepts **399 out of 440 delays (90.68% recall)** with 338 false alerts, cutting missed delays from 440 down to 41.
- **Policy C (Interval-Aware ETA)**: Shows a range (e.g. 35–45 min) whenever uncertainty width $\ge 13$ min, managing customer expectations upfront.
- **Policy D (Operations Queue)**: Routes the top 10% highest risk orders to a dispatcher queue for driver reassignment or restaurant prep expediting."

---

## 4. Key Metrics Quick Summary Table for Viva
| Metric Name | Value | Meaning |
|---|---|---|
| **Total Operational Orders** | 45,584 | Real logistics delivery dataset |
| **Chronological Test Orders** | 6,838 | Held-out test set evaluated forward in time |
| **Baseline Global Median MAE** | 7.48 min | Simple 26-minute fixed guess |
| **Best Model Test MAE** | **3.26 min** | HistGradientBoosting (56.4% error reduction) |
| **Best Model Test RMSE** | **4.08 min** | Measures variance and large error penalty |
| **Best Model Test $R^2$** | **0.8080** | Explains 80.8% of variance in delivery duration |
| **Nominal Interval Coverage** | 90.0% | Target statistical confidence |
| **Conformal Observed Coverage** | **90.33%** | Empirical verification of finite-sample coverage |
| **Quantile GB Observed Coverage**| **87.34%** | Empirical coverage with adaptive width (12.7 min) |
| **Promise-Break Base Rate** | 6.87% | Ground truth percentage of delayed orders |
| **Promise Risk ROC-AUC** | **0.9829** | Discrimination power of delay classifier |
| **Promise Risk PR-AUC** | **0.8053** | Precision-Recall curve on imbalanced late class |
| **Delay Recall at $T^* = 0.57$** | **90.68%** | Caught 399 out of 440 test delayed orders early |
| **FoodHub Satisfaction ROC-AUC**| **0.4615** | Demonstrates duration alone cannot predict ratings |
