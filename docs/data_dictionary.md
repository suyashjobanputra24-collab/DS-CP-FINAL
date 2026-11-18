# PromiseGuard: Comprehensive Data Dictionary & Schema Specification

**Platform**: PromiseGuard™ Enterprise v3.2.0  
**Target Fleet Implementations**: Zomato, Swiggy, Zepto, Blinkit, DoorDash  
**Datasets**:
1. Operational Delivery Dataset (`data/raw/delivery_operations.csv`)
2. Customer Experience Dataset (`data/raw/foodhub_order.csv`)

---

## 1. Operational Delivery Dataset (`delivery_operations.csv`)

| Column Name | Data Type | Units / Allowed Values | Description | Timing / Availability |
|---|---|---|---|---|
| `ID` | String | Alphanumeric (e.g. `0xcdcd`) | Unique anonymized order transaction identifier | Order Placement |
| `Delivery_person_ID` | String | String prefix + ID (e.g. `DEHRES17DEL01`) | Unique delivery courier identifier (prefix encodes city) | Courier Assignment |
| `Delivery_person_Age` | Float | 20.0 to 50.0 years | Age of the delivery partner | Order Placement |
| `Delivery_person_Ratings` | Float | 1.0 to 5.0 (Historical) | Historical average performance rating of the courier | Order Placement |
| `Restaurant_latitude` | Float | Decimal degrees | Latitude of the restaurant location | Order Placement |
| `Restaurant_longitude` | Float | Decimal degrees | Longitude of the restaurant location | Order Placement |
| `Delivery_location_latitude`| Float | Decimal degrees | Latitude of the customer delivery drop-off point | Order Placement |
| `Delivery_location_longitude`| Float | Decimal degrees | Longitude of the customer delivery drop-off point | Order Placement |
| `Order_Date` | String | `DD-MM-YYYY` | Date when the food order was placed | Order Placement |
| `Time_Orderd` | String | `HH:MM` | Timestamp when order was placed by customer | Order Placement |
| `Time_Order_picked` | String | `HH:MM` | Timestamp when courier picked up food from restaurant | Order Pickup |
| `Weather_conditions` | Categorical | `Sunny`, `Cloudy`, `Windy`, `Fog`, `Stormy`, `Sandstorms` | Atmospheric weather condition during order window | Order Placement |
| `Road_traffic_density` | Categorical | `Low`, `Medium`, `High`, `Jam` | Traffic density category along route | Order Placement |
| `Vehicle_condition` | Integer | `0` (Poor), `1` (Standard), `2` (Good) | Maintenance and condition tier of delivery vehicle | Order Placement |
| `Type_of_order` | Categorical | `Meal`, `Snack`, `Drinks`, `Buffet` | Food package volume/type classification | Order Placement |
| `Type_of_vehicle` | Categorical | `motorcycle`, `scooter`, `electric_scooter`, `bicycle` | Mode of transport used by delivery courier | Order Placement |
| `multiple_deliveries` | Float | `0.0`, `1.0`, `2.0`, `3.0` | Number of concurrent orders batched to the courier | Order Placement |
| `Festival` | Binary | `No`, `Yes` | Indicator of whether order occurred during a cultural festival | Order Placement |
| `City` | Categorical | `Metropolitian`, `Urban`, `Semi-Urban` | Urban density classification of delivery zone | Order Placement |
| `Time_taken (min)` | Integer | 10 to 54 minutes | **Primary Target**: Total duration from placement to delivery | **Post-Delivery Target** |

### Engineered Features
| Feature Name | Derived From | Formula / Description | Role |
|---|---|---|---|
| `distance_km` | Restaurant & Delivery GPS | Great-circle Haversine formula on absolute coordinates | Predictor Feature |
| `order_hour` | `Time_Orderd` | Hour of order placement (0 to 23) | Predictor Feature |
| `order_day_of_week`| `Order_Date` | Day of week (`Monday` to `Sunday`) | Predictor Feature |
| `is_weekend` | `Order_Date` | `1` if Saturday or Sunday, else `0` | Predictor Feature |
| `is_peak_hour` | `order_hour` | `1` if hour in [12-14, 19-22], else `0` | Predictor Feature |
| `pickup_prep_delay_min`| `Time_Order_picked` - `Time_Orderd` | Minutes courier waited for kitchen packaging | Predictor Feature |
| `promised_eta_min` | `distance_km`, `Road_traffic_density` | Operational commitment SLA ($15.0 + 2.2 \cdot d + \text{Buffer}$) | Promise Boundary |
| `promise_error_min`| `Time_taken (min)` - `promised_eta_min` | Actual delivery duration minus promised SLA | Reliability Metric |
| `is_promise_broken`| `promise_error_min` | `1` if $\text{promise\_error\_min} > 0$, else `0` | Classification Target |

---

## 2. Customer Satisfaction Dataset (`foodhub_order.csv`)

| Column Name | Data Type | Units / Allowed Values | Description | Role |
|---|---|---|---|---|
| `order_id` | Integer | Unique integer | Anonymized order identifier | Primary ID |
| `customer_id` | Integer | Unique integer | Customer account identifier | Identifier |
| `restaurant_name` | String | Restaurant name string | Name of fulfilling food outlet | Categorical Feature |
| `cuisine_type` | Categorical | American, Japanese, Italian, etc. | Culinary food style | Categorical Feature |
| `cost_of_the_order`| Float | US Dollars ($) | Total transaction cost of food items | Financial Feature |
| `day_of_the_week` | Categorical | `Weekday`, `Weekend` | Day type of order | Temporal Feature |
| `rating` | String / Numeric | `3`, `4`, `5`, or `'Not given'` | Genuine post-delivery customer rating | Customer Satisfaction |
| `food_preparation_time` | Integer | 20 to 35 minutes | Time spent in kitchen preparing meal | Operational Stage |
| `delivery_time` | Integer | 15 to 33 minutes | Time courier spent in transit delivering order | Operational Stage |
| `total_order_time` | Integer | `food_preparation_time` + `delivery_time` | Total order turnaround fulfillment duration | Derived Operational Time |
| `is_rated` | Binary | `1` if rating != 'Not given', else `0` | Rating presence indicator | Response Filter |
| `is_satisfied` | Binary | `1` if rating $\ge 4$, `0` if rating $\le 3$ | Binarized customer satisfaction label | Classification Target |
