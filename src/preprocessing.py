"""
PromiseGuard: Preprocessing and Feature Engineering Pipeline
Handles coordinate cleaning, Haversine distance, temporal features,
imputation, and chronological train/validation/test splitting.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Computes great circle distance between two points on earth in kilometers.
    """
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c


class DeliveryDataPipeline:
    def __init__(self, raw_path: str = "data/raw/delivery_operations.csv"):
        self.raw_path = raw_path
        self.processed_dir = "data/processed"
        os.makedirs(self.processed_dir, exist_ok=True)

    def load_and_clean_raw(self) -> pd.DataFrame:
        df = pd.read_csv(self.raw_path)
        
        # Strip string columns
        str_cols = df.select_dtypes(include=["object", "string"]).columns
        for c in str_cols:
            df[c] = df[c].astype(str).str.strip()
            # Replace 'nan', 'NaN', 'None', '' with actual np.nan
            df[c] = df[c].replace({"nan": np.nan, "NaN": np.nan, "None": np.nan, "": np.nan})

        # Coordinates cleaning: Take absolute values to correct GPS sign flipping
        df["res_lat_clean"] = df["Restaurant_latitude"].abs()
        df["res_lon_clean"] = df["Restaurant_longitude"].abs()
        df["del_lat_clean"] = df["Delivery_location_latitude"].abs()
        df["del_lon_clean"] = df["Delivery_location_longitude"].abs()

        # Valid coordinates mask (India roughly 5°N to 38°N, 65°E to 98°E)
        valid_coords = (
            (df["res_lat_clean"] > 5) & (df["res_lat_clean"] < 40) &
            (df["res_lon_clean"] > 60) & (df["res_lon_clean"] < 100) &
            (df["del_lat_clean"] > 5) & (df["del_lat_clean"] < 40) &
            (df["del_lon_clean"] > 60) & (df["del_lon_clean"] < 100)
        )
        df["valid_coordinates"] = valid_coords

        # Compute Haversine Distance
        df["distance_km"] = haversine_distance(
            df["res_lat_clean"], df["res_lon_clean"],
            df["del_lat_clean"], df["del_lon_clean"]
        )

        # For invalid coordinates, impute distance with median distance
        median_dist = df.loc[valid_coords, "distance_km"].median()
        df["distance_km"] = df["distance_km"].fillna(median_dist)
        df.loc[~valid_coords, "distance_km"] = median_dist

        # Parse Order_Date: e.g. "12-02-2022" (day-month-year)
        df["order_datetime"] = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y", errors="coerce")
        df["order_day_of_week"] = df["order_datetime"].dt.day_name()
        df["is_weekend"] = df["order_datetime"].dt.dayofweek.isin([5, 6]).astype(int)

        # Parse Order Time and Pickup Time
        def parse_minutes(time_val):
            if pd.isna(time_val):
                return np.nan
            try:
                parts = str(time_val).strip().split(":")
                return int(parts[0]) * 60 + int(parts[1])
            except:
                return np.nan

        df["order_time_min"] = df["Time_Orderd"].apply(parse_minutes)
        df["pickup_time_min"] = df["Time_Order_picked"].apply(parse_minutes)

        # Pickup delay (prep / dispatch buffer)
        df["pickup_prep_delay_min"] = df["pickup_time_min"] - df["order_time_min"]
        # Handle midnight crossover if any
        df.loc[df["pickup_prep_delay_min"] < 0, "pickup_prep_delay_min"] += 1440
        # If order_time_min is null, impute pickup delay with median (10 min)
        df["pickup_prep_delay_min"] = df["pickup_prep_delay_min"].fillna(10.0)
        # Constrain reasonable prep delay between 2 and 30 mins
        df.loc[(df["pickup_prep_delay_min"] > 30) | (df["pickup_prep_delay_min"] < 1), "pickup_prep_delay_min"] = 10.0

        # Order hour
        df["order_hour"] = (df["order_time_min"] // 60).fillna(df["pickup_time_min"] // 60)
        df["order_hour"] = df["order_hour"].fillna(df["order_hour"].median())

        # Peak hours: Lunch (12-14) & Dinner (19-22)
        df["is_peak_hour"] = df["order_hour"].isin([12, 13, 14, 19, 20, 21, 22]).astype(int)

        # Extract City code prefix from Delivery_person_ID
        df["city_code"] = df["Delivery_person_ID"].str[:3]

        # Clean multiple_deliveries (fill missing with 0)
        df["multiple_deliveries"] = pd.to_numeric(df["multiple_deliveries"], errors="coerce").fillna(0.0)

        # Clean courier age & rating with training-safe medians
        df["Delivery_person_Age"] = pd.to_numeric(df["Delivery_person_Age"], errors="coerce")
        df["Delivery_person_Ratings"] = pd.to_numeric(df["Delivery_person_Ratings"], errors="coerce")

        # Categoricals fillna with mode/unknown
        df["Weather_conditions"] = df["Weather_conditions"].fillna("Sunny")
        df["Road_traffic_density"] = df["Road_traffic_density"].fillna("Medium")
        df["City"] = df["City"].fillna("Metropolitian")
        df["Festival"] = df["Festival"].fillna("No")

        # Delivery promise SLA definition
        # Industry rule: Baseline promise = 15 min base prep/handover + (distance_km * 2.2 min/km) + traffic_buffer
        traffic_buffer_map = {"Low": 2.0, "Medium": 6.0, "High": 12.0, "Jam": 18.0}
        traffic_buf = df["Road_traffic_density"].map(traffic_buffer_map).fillna(6.0)
        df["promised_eta_min"] = np.round(15.0 + (df["distance_km"] * 2.2) + traffic_buf, 1)

        # Promise error and late flag
        df["promise_error_min"] = df["Time_taken (min)"] - df["promised_eta_min"]
        df["is_promise_broken"] = (df["promise_error_min"] > 0).astype(int)

        return df

    def prepare_and_save_data(self):
        df = self.load_and_clean_raw()
        
        # Sort chronologically by Order_Date and Time_Orderd for strict temporal split
        df_sorted = df.sort_values(by=["order_datetime", "order_time_min"]).reset_index(drop=True)

        n = len(df_sorted)
        train_idx = int(n * 0.70)
        val_idx = int(n * 0.85)

        train_df = df_sorted.iloc[:train_idx].copy()
        val_df = df_sorted.iloc[train_idx:val_idx].copy()
        test_df = df_sorted.iloc[val_idx:].copy()

        train_df.to_csv(os.path.join(self.processed_dir, "train_delivery.csv"), index=False)
        val_df.to_csv(os.path.join(self.processed_dir, "val_delivery.csv"), index=False)
        test_df.to_csv(os.path.join(self.processed_dir, "test_delivery.csv"), index=False)
        df_sorted.to_csv(os.path.join(self.processed_dir, "cleaned_delivery_operations.csv"), index=False)

        print(f"Data pipeline complete! Saved to {self.processed_dir}/")
        print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")
        return train_df, val_df, test_df


class FoodHubPipeline:
    def __init__(self, raw_path: str = "data/raw/foodhub_order.csv"):
        self.raw_path = raw_path
        self.processed_dir = "data/processed"
        os.makedirs(self.processed_dir, exist_ok=True)

    def load_and_clean_raw(self) -> pd.DataFrame:
        df = pd.read_csv(self.raw_path)
        
        # Clean ratings
        df["rating_raw"] = df["rating"]
        df["is_rated"] = (df["rating"] != "Not given").astype(int)
        df["rating_numeric"] = pd.to_numeric(df["rating"], errors="coerce")

        # Total order delivery turnaround time
        df["total_order_time"] = df["food_preparation_time"] + df["delivery_time"]

        # Satisfaction indicator (for rated orders: 4-5 = Satisfied (1), 1-3 = Dissatisfied (0))
        df["is_satisfied"] = np.nan
        df.loc[df["rating_numeric"] >= 4, "is_satisfied"] = 1
        df.loc[df["rating_numeric"] < 4, "is_satisfied"] = 0

        # Save cleaned
        clean_path = os.path.join(self.processed_dir, "cleaned_foodhub_orders.csv")
        df.to_csv(clean_path, index=False)
        print(f"FoodHub pipeline complete! Saved to {clean_path}")
        return df


if __name__ == "__main__":
    ops_pipe = DeliveryDataPipeline()
    ops_pipe.prepare_and_save_data()

    fh_pipe = FoodHubPipeline()
    fh_pipe.load_and_clean_raw()
