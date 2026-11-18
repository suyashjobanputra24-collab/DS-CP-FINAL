# PromiseGuard: Industry Context and Literature Review

**Course**: MD3135 Data Science | Third Year Computer Engineering  
**Subject Area**: Delivery Logistics, Machine Learning, Uncertainty Estimation & Explainability  

---

## 1. Public Engineering Disclosures by Food Delivery Aggregators

### 1.1 Zomato Engineering: "The Accurate ETA to Customer Satisfaction"
Zomato has published extensive engineering blogs outlining the components of its real-world ETA architecture:
1. **Multi-Stage Decomposition**:
   - Total ETA is decomposed into **Kitchen Preparation Time (KPT)**, **Courier Assignment Time**, **First-Mile Travel (Courier to Restaurant)**, **Courier Waiting Time at Restaurant**, and **Last-Mile Travel (Restaurant to Customer)**.
2. **Contextual Inputs**:
   - Dynamic buffers derived from localized partner availability, order size, dish complexity, historical restaurant delays, weather disruptions (monsoon rain models), and road congestion indices.
3. **Specialized Rain & Quantile Models**:
   - Zomato publicly described implementing specialized models to handle extreme late deliveries and weather disruptions, shifting from single point estimates toward higher quantile estimation to prevent severe late surprises during rain.

### 1.2 Swiggy: Logistics Intelligence & Multi-Service Dispatch
Swiggy operates proprietary hyper-local dispatch engines that perform:
1. **Batching Algorithms**: Assigning multiple concurrent orders to the same courier when delivery routes have minimal geographic detour and kitchen ready times align.
2. **Dynamic Kitchen Readiness**: Incorporating IoT restaurant merchant tablets where kitchens mark food as "food ready" to optimize the timing of courier arrival.

### 1.3 Meituan: Real-Time Intelligent Dispatching
Meituan operates one of the world's largest minute-level on-demand delivery networks in China. Published literature (e.g. INFORMS TSL Challenge and INFORMS Interfaces 2023) describes:
1. Combining Operations Research (stochastic vehicle routing, combinatorial assignment) with Reinforcement Learning and Deep Learning to dispatch orders dynamically under severe urban uncertainties.
2. Demonstrating that ETA prediction accuracy directly influences dispatch optimization, courier safety, and customer satisfaction.

---

## 2. Positioning PromiseGuard Honestly
| Capability | Live Commercial Platforms (Zomato/Swiggy) | PromiseGuard Academic Prototype |
|---|---|---|
| **Data Flow** | Real-time streaming (Kafka, Flink, live GPS) | Chronological batch processing ($N = 45,584$ test splits) |
| **ETA Output** | Rolling dynamic minutes updated per second | Point ETA + Calibrated 90% Conformal & Quantile Intervals |
| **Delay Management**| Proprietary real-time re-dispatching | Offline policy simulation comparing alert thresholds ($T^*$) |
| **Explainability** | Internal feature telemetry & monitoring | Transparent global & local SHAP explanations |
| **Satisfaction** | Millions of live in-app user ratings & chats | Empirical study of genuine customer ratings (FoodHub) |

**Conclusion**: PromiseGuard does not claim to replace production dispatch engines. It provides an open, academically rigorous, reproducible framework demonstrating how data science techniques combine to analyze prediction reliability, uncertainty calibration, and decision policies.
