# PromiseGuard: Comprehensive Data Audit Report

**Date of Audit**: November 2026  
**Auditor**: PromiseGuard Data Governance Pipeline  
**Datasets Audited**:
1. `Delivery Operations (Zomato Benchmark)`
2. `FoodHub Customer Experience`

---

## 1. Primary Operational Dataset: Delivery Operations (Zomato Benchmark)

- **Total Observations (Rows)**: 45,584
- **Total Variables (Columns)**: 20
- **Duplicate Records**: 0
- **Unique Order IDs**: 45,584
- **Unique Delivery Partners**: 1,320

### Missingness Profile
| Column | Missing Count | Missing % | Status |
|---|---|---|---|
| `ID` | 0 | 0.0% | Clean |
| `Delivery_person_ID` | 0 | 0.0% | Clean |
| `Delivery_person_Age` | 1,854 | 4.07% | Low (<5%) |
| `Delivery_person_Ratings` | 1,908 | 4.19% | Low (<5%) |
| `Restaurant_latitude` | 0 | 0.0% | Clean |
| `Restaurant_longitude` | 0 | 0.0% | Clean |
| `Delivery_location_latitude` | 0 | 0.0% | Clean |
| `Delivery_location_longitude` | 0 | 0.0% | Clean |
| `Order_Date` | 0 | 0.0% | Clean |
| `Time_Orderd` | 1,731 | 3.8% | Low (<5%) |
| `Time_Order_picked` | 0 | 0.0% | Clean |
| `Weather_conditions` | 616 | 1.35% | Low (<5%) |
| `Road_traffic_density` | 601 | 1.32% | Low (<5%) |
| `Vehicle_condition` | 0 | 0.0% | Clean |
| `Type_of_order` | 0 | 0.0% | Clean |
| `Type_of_vehicle` | 0 | 0.0% | Clean |
| `multiple_deliveries` | 993 | 2.18% | Low (<5%) |
| `Festival` | 228 | 0.5% | Low (<5%) |
| `City` | 1,200 | 2.63% | Low (<5%) |
| `Time_taken (min)` | 0 | 0.0% | Clean |

### Key Numeric Distributions
| Variable | Min | Mean | Median | Max | Std Dev |
|---|---|---|---|---|---|
| `Delivery_person_Age` | 15.0 | 29.57 | 30.0 | 50.0 | 5.82 |
| `Delivery_person_Ratings` | 1.0 | 4.63 | 4.7 | 6.0 | 0.33 |
| `Restaurant_latitude` | -30.905562 | 17.02 | 18.55144 | 30.914057 | 8.19 |
| `Restaurant_longitude` | -88.366217 | 70.23 | 75.897963 | 88.433452 | 22.89 |
| `Delivery_location_latitude` | 0.01 | 17.47 | 18.633934 | 31.054057 | 7.34 |
| `Delivery_location_longitude` | 0.01 | 70.84 | 76.002574 | 88.563452 | 21.12 |
| `Vehicle_condition` | 0.0 | 1.02 | 1.0 | 3.0 | 0.84 |
| `multiple_deliveries` | 0.0 | 0.74 | 1.0 | 3.0 | 0.57 |
| `Time_taken (min)` | 10.0 | 26.29 | 26.0 | 54.0 | 9.38 |

---

## 2. Customer Satisfaction Dataset: FoodHub Customer Experience

- **Total Observations (Rows)**: 1,898
- **Total Variables (Columns)**: 9
- **Duplicate Records**: 0
- **Unique Orders**: 1,898
- **Unique Customers**: 1,200

### Missingness Profile
| Column | Missing Count | Missing % | Status |
|---|---|---|---|
| `order_id` | 0 | 0.0% | Clean |
| `customer_id` | 0 | 0.0% | Clean |
| `restaurant_name` | 0 | 0.0% | Clean |
| `cuisine_type` | 0 | 0.0% | Clean |
| `cost_of_the_order` | 0 | 0.0% | Clean |
| `day_of_the_week` | 0 | 0.0% | Clean |
| `rating` | 0 | 0.0% | Clean |
| `food_preparation_time` | 0 | 0.0% | Clean |
| `delivery_time` | 0 | 0.0% | Clean |

### Key Numeric Distributions
| Variable | Min | Mean | Median | Max | Std Dev |
|---|---|---|---|---|---|
| `order_id` | 1476547.0 | 1477495.5 | 1477495.5 | 1478444.0 | 548.05 |
| `customer_id` | 1311.0 | 171168.48 | 128600.0 | 405334.0 | 113698.14 |
| `cost_of_the_order` | 4.47 | 16.5 | 14.14 | 35.41 | 7.48 |
| `food_preparation_time` | 20.0 | 27.37 | 27.0 | 35.0 | 4.63 |
| `delivery_time` | 15.0 | 24.16 | 25.0 | 33.0 | 4.97 |

---

## 3. Data Governance & Integrity Notes

1. **Delivery Partner Ratings vs. Customer Satisfaction**:
   - In `delivery_operations.csv`, `Delivery_person_Ratings` reflects the courier's historical rating, NOT the customer's satisfaction rating for that specific order.
   - Genuine order-level customer satisfaction is evaluated using `foodhub_order.csv`, where `rating` represents the customer's explicit post-delivery rating.

2. **Unrated Orders ('Not given')**:
   - In `foodhub_order.csv`, 736 orders (38.78%) have `rating == 'Not given'`.
   - In accordance with non-negotiable research integrity rules, these orders are NOT imputed as satisfied or dissatisfied. They are analyzed as an explicit 'Unrated' category.

3. **Geographic Coordinates Integrity**:
   - In `delivery_operations.csv`, some coordinates contain sign inversions or zero-readings from sensor dropouts. Preprocessing corrects absolute values and validates sensible ranges.
