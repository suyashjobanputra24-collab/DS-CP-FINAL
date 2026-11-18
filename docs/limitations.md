# PromiseGuard: Academic Limitations, Ethics & Governance Safeguards

**Course**: MD3135 Data Science | Third Year Computer Engineering  
**Project**: PromiseGuard: An Explainable Machine Learning Framework  
**Authors**: Suyash Jobanputra & Shreyash Jobanputra  

---

## 1. Academic & Research Limitations

1. **Static Benchmark vs Real-Time Streaming**:
   - PromiseGuard operates as an offline batch inference and analytical framework. 
   - Real-world production platforms (such as Zomato or Swiggy) employ streaming message brokers (Kafka/Flink) to continuously update ETAs dynamically as a courier progresses along their route.

2. **Absence of Stated Customer Promises in Historical Logs**:
   - The operational delivery dataset records actual delivery duration and courier/order context, but does not capture the exact dynamic countdown seconds displayed on an individual customer's mobile app screen.
   - We derived an explicit operational SLA promise boundary based on industry baseline rules ($15\text{ min} + 2.2 \cdot d + \text{traffic buffer}$) to study promise-break risk rigorously.

3. **Customer Satisfaction Voluntary Response Bias**:
   - In the FoodHub dataset, 38.78% of orders had unrated feedback (`'Not given'`).
   - Customers typically leave ratings when experiences are either exceptionally pleasant or deeply frustrating, leading to bimodal voluntary response bias. Imputing unrated orders as dissatisfied would violate data integrity.

4. **Weak Direct Coupling Between Duration and Star Ratings**:
   - Our empirical findings show that delivery transit duration alone is weakly associated with post-delivery ratings ($\rho = -0.01$, ROC-AUC $\approx 0.46$). 
   - A customer whose food arrives on time but is cold, spilled, or missing cutlery will leave a 1-star rating regardless of high courier speed. Conversely, a patient customer may leave 5 stars for delicious food even after a 10-minute transit delay.

5. **Attribution vs. Causality in Explainable ML**:
   - SHAP values quantify mathematical feature importance inside the tree ensemble.
   - They do **NOT** prove real-world causality. For example, high traffic often coincides with peak mealtime demand and restaurant kitchen backlogs; attributing the full delay to road conditions alone is a confounded attribution.

---

## 2. Ethical Safeguards & Worker Fairness

1. **Delivery Partner Safety**:
   - ETA systems must never be used to exert punitive pressure or encourage unsafe driving, speeding, or red-light violations.
   - PromiseGuard's proactive intervention recommendations focus strictly on **customer expectation management** (sending realistic revised ETA ranges) and **kitchen dispatch review**, never on driver penalties.

2. **Fairness Across Urban Tiers**:
   - High prediction errors in semi-urban or congested neighborhoods often reflect poor road infrastructure, unmapped lanes, or cellular connectivity dead zones rather than driver negligence.
   - Algorithmic decisions must not penalize couriers operating in topographically difficult delivery sectors.

3. **Automation Bias**:
   - A delay risk score is an operational decision-support recommendation, not an immutable directive. Human dispatch coordinators must always retain manual override capabilities.
