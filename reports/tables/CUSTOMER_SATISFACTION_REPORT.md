# PromiseGuard: Customer Satisfaction & Dissatisfaction Analytics Report

**Primary Customer Dataset**: FoodHub Real-World Order Experience  
**Total Observations**: 1,898 orders  
**Rated Observations**: 1,162 (61.2%) | **Unrated Orders ('Not given')**: 736 (38.78%)  

---

## 1. Customer Rating Profile
- **5 Stars (Highly Satisfied)**: 588 orders (31.0%)
- **4 Stars (Satisfied)**: 386 orders (20.3%)
- **3 Stars (Dissatisfied/Neutral)**: 188 orders (9.9%)
- **Unrated ('Not given')**: 736 orders (38.78%)

---

## 2. Transit & Preparation Time vs Customer Rating

| Customer Rating | Mean Delivery Time (min) | Median Delivery Time (min) | Mean Food Prep Time (min) | Total Turnaround Time (min) |
|---|---|---|---|---|
| **Rating 5 (Top)** | 24.16 min | 25.0 min | 27.42 min | 51.58 min |
| **Rating 4 (Good)** | 24.19 min | 25.0 min | 27.24 min | 51.43 min |
| **Rating 3 (Low)** | 24.64 min | 25.0 min | 27.34 min | 51.98 min |

---

## 3. Machine Learning Dissatisfaction Classifier

| Classifier Architecture | Test ROC-AUC | Recall on Dissatisfied | Precision on Dissatisfied | Overall Accuracy |
|---|---|---|---|---|
| **Logistic Regression (Balanced)** | **0.4615** | 40.4% | 13.5% | 48.4% |
| **Random Forest (Balanced)** | **0.4331** | 31.9% | 12.4% | 52.6% |

---

## 4. Key Academic Insights & Limitations

1. **The Unrated Order Dilemma**:
   - Over 38% of customers do not leave a rating. In standard food delivery analytics, satisfied customers frequently skip rating unless incentivized, creating severe **voluntary response bias**.
   - Imputing missing ratings as either satisfied or dissatisfied would distort the empirical distribution.

2. **Weak Direct Coupling with Duration Alone**:
   - The difference in mean delivery time between 5-star ratings (24.16 min) and 3-star ratings (24.64 min) is under 30 seconds.
   - This proves that delivery duration by itself is NOT a deterministic predictor of customer satisfaction. Factors such as food temperature, order accuracy, missing cutlery/condiments, packaging integrity, and taste play overwhelming roles in customer perception.

3. **Appropriate Modeling Scope**:
   - While delivery delay contributes to negative experiences, an ML model trained exclusively on transit times cannot reliably classify satisfaction. This negative result is scientifically honest and highlights why platforms must gather multi-dimensional feedback.
