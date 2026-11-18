"""
PromiseGuard: Exploratory Data Analysis (EDA) Module
Generates all 19 research-question-driven visualisations for delivery operations
and customer satisfaction analytics. Saves publication-grade figures to reports/figures/.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["figure.dpi"] = 300


class EDAPipeline:
    def __init__(self, data_dir: str = "data/processed", fig_dir: str = "reports/figures"):
        self.data_dir = data_dir
        self.fig_dir = fig_dir
        os.makedirs(self.fig_dir, exist_ok=True)

        self.ops_df = pd.read_csv(os.path.join(data_dir, "cleaned_delivery_operations.csv"))
        self.fh_df = pd.read_csv(os.path.join(data_dir, "cleaned_foodhub_orders.csv"))
        self.primary_color = "#1E3A8A"  # Deep Navy Blue
        self.secondary_color = "#D97706"  # Amber Orange
        self.accent_color = "#10B981"  # Emerald Green
        self.danger_color = "#EF4444"  # Crimson Red

    def generate_all_plots(self):
        print("Generating EDA Visualizations...")
        self.plot_01_delivery_time_distribution()
        self.plot_02_distance_vs_delivery_time()
        self.plot_03_traffic_density_boxplots()
        self.plot_04_weather_conditions_distribution()
        self.plot_05_hourly_delivery_patterns()
        self.plot_06_weekday_vs_weekend()
        self.plot_07_vehicle_type_performance()
        self.plot_08_multiple_deliveries_impact()
        self.plot_09_prep_buffer_distribution()
        self.plot_10_correlation_heatmap()
        self.plot_11_missing_data_profile()
        self.plot_12_outlier_boxplot_analysis()
        self.plot_13_foodhub_rating_distribution()
        self.plot_14_foodhub_delivery_time_by_rating()
        self.plot_15_foodhub_prep_time_by_rating()
        self.plot_16_foodhub_total_order_time_distribution()
        self.plot_17_city_level_duration_distribution()
        self.plot_18_courier_ratings_vs_duration()
        self.plot_19_promise_error_distribution()
        print(f"All 19 EDA figures successfully generated in {self.fig_dir}/")

    def _save_fig(self, filename: str):
        plt.tight_layout()
        path = os.path.join(self.fig_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_01_delivery_time_distribution(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(self.ops_df["Time_taken (min)"], bins=30, kde=True, color=self.primary_color, ax=ax)
        mean_val = self.ops_df["Time_taken (min)"].mean()
        median_val = self.ops_df["Time_taken (min)"].median()
        ax.axvline(mean_val, color=self.danger_color, linestyle="--", label=f"Mean: {mean_val:.1f} min")
        ax.axvline(median_val, color=self.secondary_color, linestyle="-", label=f"Median: {median_val:.1f} min")
        ax.set_title(f"Delivery Duration Distribution (N = {len(self.ops_df):,})")
        ax.set_xlabel("Delivery Time (minutes)")
        ax.set_ylabel("Order Count")
        ax.legend()
        self._save_fig("01_delivery_time_distribution.png")

    def plot_02_distance_vs_delivery_time(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        valid_df = self.ops_df[self.ops_df["valid_coordinates"]].sample(n=min(5000, len(self.ops_df)), random_state=42)
        sns.regplot(
            data=valid_df, x="distance_km", y="Time_taken (min)",
            scatter_kws={"alpha": 0.15, "color": self.primary_color, "s": 15},
            line_kws={"color": self.danger_color, "linewidth": 2},
            ax=ax
        )
        ax.set_title("Delivery Distance vs Total Delivery Duration (Sample N = 5,000)")
        ax.set_xlabel("Haversine Distance (kilometers)")
        ax.set_ylabel("Delivery Time (minutes)")
        self._save_fig("02_distance_vs_delivery_time.png")

    def plot_03_traffic_density_boxplots(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        traffic_order = ["Low", "Medium", "High", "Jam"]
        sns.boxplot(
            data=self.ops_df, x="Road_traffic_density", y="Time_taken (min)",
            order=traffic_order, palette="Blues_r", ax=ax
        )
        ax.set_title(f"Delivery Duration Across Road Traffic Densities (N = {len(self.ops_df):,})")
        ax.set_xlabel("Traffic Density Condition")
        ax.set_ylabel("Delivery Time (minutes)")
        self._save_fig("03_traffic_density_boxplots.png")

    def plot_04_weather_conditions_distribution(self):
        fig, ax = plt.subplots(figsize=(9, 5))
        weather_order = self.ops_df.groupby("Weather_conditions")["Time_taken (min)"].median().sort_values().index
        sns.boxplot(
            data=self.ops_df, x="Weather_conditions", y="Time_taken (min)",
            order=weather_order, palette="YlOrRd", ax=ax
        )
        ax.set_title(f"Delivery Duration Across Weather Conditions (N = {len(self.ops_df):,})")
        ax.set_xlabel("Weather Condition")
        ax.set_ylabel("Delivery Time (minutes)")
        self._save_fig("04_weather_conditions_distribution.png")

    def plot_05_hourly_delivery_patterns(self):
        fig, ax = plt.subplots(figsize=(9, 5))
        hourly_stats = self.ops_df.groupby("order_hour")["Time_taken (min)"].agg(["mean", "median", "count"]).reset_index()
        ax.plot(hourly_stats["order_hour"], hourly_stats["mean"], marker="o", color=self.primary_color, label="Mean Duration")
        ax.plot(hourly_stats["order_hour"], hourly_stats["median"], marker="s", linestyle="--", color=self.secondary_color, label="Median Duration")
        ax.set_title("Diurnal Pattern: Delivery Duration by Order Hour")
        ax.set_xlabel("Order Placement Hour of Day (0–23)")
        ax.set_ylabel("Delivery Time (minutes)")
        ax.set_xticks(range(int(hourly_stats["order_hour"].min()), int(hourly_stats["order_hour"].max()) + 1))
        ax.legend()
        self._save_fig("05_hourly_delivery_patterns.png")

    def plot_06_weekday_vs_weekend(self):
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.violinplot(
            data=self.ops_df, x="is_weekend", y="Time_taken (min)",
            palette=["#3B82F6", "#F59E0B"], ax=ax
        )
        ax.set_xticklabels(["Weekday (Mon-Fri)", "Weekend (Sat-Sun)"])
        ax.set_title("Delivery Duration: Weekday vs Weekend Comparison")
        ax.set_xlabel("Day Category")
        ax.set_ylabel("Delivery Time (minutes)")
        self._save_fig("06_weekday_vs_weekend_comparison.png")

    def plot_07_vehicle_type_performance(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=self.ops_df, x="Type_of_vehicle", y="Time_taken (min)",
            estimator=np.mean, errorbar="sd", palette="viridis", ax=ax
        )
        ax.set_title("Mean Delivery Duration by Vehicle Type (with Std Dev)")
        ax.set_xlabel("Vehicle Category")
        ax.set_ylabel("Mean Delivery Time (minutes)")
        self._save_fig("07_vehicle_type_performance.png")

    def plot_08_multiple_deliveries_impact(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=self.ops_df, x="multiple_deliveries", y="Time_taken (min)",
            palette="Purples", ax=ax
        )
        ax.set_title("Impact of Multiple Batch Deliveries on Order Duration")
        ax.set_xlabel("Number of Concurrently Assigned Orders")
        ax.set_ylabel("Delivery Time (minutes)")
        self._save_fig("08_multiple_deliveries_impact.png")

    def plot_09_prep_buffer_distribution(self):
        fig, ax = plt.subplots(figsize=(7, 5))
        val_counts = self.ops_df["pickup_prep_delay_min"].value_counts().sort_index()
        bars = ax.bar(val_counts.index.astype(str), val_counts.values, color=self.primary_color, width=0.5)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 300, f"{yval:,}", ha="center", va="bottom", fontsize=9)
        ax.set_title("Restaurant Prep / Pickup Delay Buffer Distribution")
        ax.set_xlabel("Pickup Delay (minutes)")
        ax.set_ylabel("Order Frequency")
        self._save_fig("09_prep_buffer_distribution.png")

    def plot_10_correlation_heatmap(self):
        fig, ax = plt.subplots(figsize=(9, 7))
        num_cols = ["Time_taken (min)", "distance_km", "Delivery_person_Age", "Delivery_person_Ratings",
                    "multiple_deliveries", "pickup_prep_delay_min", "is_peak_hour", "is_weekend"]
        corr = self.ops_df[num_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-0.4, vmax=0.6, ax=ax)
        ax.set_title("Correlation Heatmap of Numerical Operational Features")
        self._save_fig("10_correlation_heatmap.png")

    def plot_11_missing_data_profile(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        raw_ops = pd.read_csv("data/raw/delivery_operations.csv")
        missing_pcts = (raw_ops.isnull().mean() * 100).sort_values(ascending=False)
        missing_pcts = missing_pcts[missing_pcts > 0]
        bars = ax.barh(missing_pcts.index, missing_pcts.values, color=self.secondary_color)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.1, bar.get_y() + bar.get_height()/2.0, f"{w:.2f}%", ha="left", va="center", fontsize=9)
        ax.set_title("Missing Data Profile in Raw Delivery Operations Dataset (%)")
        ax.set_xlabel("Missing Percentage (%)")
        ax.set_xlim(0, max(missing_pcts.values) + 1.5)
        self._save_fig("11_missing_data_profile.png")

    def plot_12_outlier_boxplot_analysis(self):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        sns.boxplot(y=self.ops_df["Time_taken (min)"], ax=axes[0], color="#93C5FD")
        axes[0].set_title("Delivery Duration Outlier Inspection")
        axes[0].set_ylabel("Delivery Time (minutes)")

        sns.boxplot(y=self.ops_df["distance_km"], ax=axes[1], color="#FDE68A")
        axes[1].set_title("Haversine Distance Outlier Inspection")
        axes[1].set_ylabel("Distance (km)")
        self._save_fig("12_outlier_boxplot_analysis.png")

    def plot_13_foodhub_rating_distribution(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        counts = self.fh_df["rating"].value_counts()
        order = ["5", "4", "3", "Not given"]
        counts = counts.reindex(order)
        colors = ["#10B981", "#3B82F6", "#F59E0B", "#9CA3AF"]
        bars = ax.bar(counts.index, counts.values, color=colors)
        for bar in bars:
            y = bar.get_height()
            pct = (y / len(self.fh_df)) * 100
            ax.text(bar.get_x() + bar.get_width()/2.0, y + 15, f"{y:,} ({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"Customer Rating Distribution in FoodHub Dataset (N = {len(self.fh_df):,})")
        ax.set_xlabel("Customer Post-Delivery Rating")
        ax.set_ylabel("Order Count")
        self._save_fig("13_foodhub_rating_distribution.png")

    def plot_14_foodhub_delivery_time_by_rating(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        rated_fh = self.fh_df[self.fh_df["is_rated"] == 1]
        sns.boxplot(
            data=rated_fh, x="rating", y="delivery_time",
            order=["3", "4", "5"], palette="Blues", ax=ax
        )
        ax.set_title(f"Delivery Time Across Customer Rating Tiers (Rated Orders N = {len(rated_fh):,})")
        ax.set_xlabel("Customer Rating")
        ax.set_ylabel("Delivery Time (minutes)")
        self._save_fig("14_foodhub_delivery_time_by_rating.png")

    def plot_15_foodhub_prep_time_by_rating(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        rated_fh = self.fh_df[self.fh_df["is_rated"] == 1]
        sns.boxplot(
            data=rated_fh, x="rating", y="food_preparation_time",
            order=["3", "4", "5"], palette="Oranges", ax=ax
        )
        ax.set_title("Food Preparation Time Across Customer Rating Tiers")
        ax.set_xlabel("Customer Rating")
        ax.set_ylabel("Preparation Time (minutes)")
        self._save_fig("15_foodhub_prep_time_by_rating.png")

    def plot_16_foodhub_total_order_time_distribution(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(self.fh_df["total_order_time"], bins=20, kde=True, color="#8B5CF6", ax=ax)
        ax.set_title(f"Total Order Duration Distribution: Prep + Delivery (N = {len(self.fh_df):,})")
        ax.set_xlabel("Total Turnaround Time (Prep + Transit in minutes)")
        ax.set_ylabel("Order Count")
        self._save_fig("16_foodhub_total_order_time_distribution.png")

    def plot_17_city_level_duration_distribution(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=self.ops_df, x="City", y="Time_taken (min)",
            palette="Pastel1", ax=ax
        )
        ax.set_title("Delivery Duration Distribution Across City Urban Tiers")
        ax.set_xlabel("City Category")
        ax.set_ylabel("Delivery Time (minutes)")
        self._save_fig("17_city_level_duration_distribution.png")

    def plot_18_courier_ratings_vs_duration(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        valid_courier = self.ops_df.dropna(subset=["Delivery_person_Ratings"]).sample(min(4000, len(self.ops_df)), random_state=42)
        sns.scatterplot(
            data=valid_courier, x="Delivery_person_Ratings", y="Time_taken (min)",
            alpha=0.2, color=self.primary_color, ax=ax
        )
        ax.set_title("Courier Historical Rating vs Delivery Duration (Sample N = 4,000)")
        ax.set_xlabel("Courier Historical Performance Rating")
        ax.set_ylabel("Delivery Time (minutes)")
        self._save_fig("18_courier_ratings_vs_duration.png")

    def plot_19_promise_error_distribution(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(self.ops_df["promise_error_min"], bins=35, kde=True, color=self.danger_color, ax=ax)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Promise Boundary (Error = 0)")
        late_pct = (self.ops_df["is_promise_broken"].mean()) * 100
        ax.text(0.05, 0.90, f"Promise-Break Rate: {late_pct:.1f}%", transform=ax.transAxes,
                fontsize=11, fontweight="bold", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_title("Operational Delivery Promise Error Distribution (Actual - Promised SLA)")
        ax.set_xlabel("Promise Error (minutes) [Positive = Delayed, Negative = Early]")
        ax.set_ylabel("Frequency")
        ax.legend()
        self._save_fig("19_promise_error_distribution.png")


if __name__ == "__main__":
    eda = EDAPipeline()
    eda.generate_all_plots()
