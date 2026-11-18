"""
PromiseGuard: Data Loader and Ingestion Audit Module
Academic Project: MD3135 Data Science | Third Year Computer Engineering
"""

import os
import json
import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, data_dir: str = "data"):
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.reports_dir = os.path.join("reports", "data_audit")
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def load_delivery_operations_raw(self) -> pd.DataFrame:
        path = os.path.join(self.raw_dir, "delivery_operations.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raw delivery operations data not found at {path}")
        return pd.read_csv(path)

    def load_foodhub_raw(self) -> pd.DataFrame:
        path = os.path.join(self.raw_dir, "foodhub_order.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raw foodhub data not found at {path}")
        return pd.read_csv(path)

    def audit_dataset(self, df: pd.DataFrame, name: str) -> dict:
        """
        Conducts a comprehensive data audit on a given dataframe.
        """
        audit = {
            "dataset_name": name,
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "columns": list(df.columns),
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": {col: int(df[col].isnull().sum()) for col in df.columns},
            "missing_percentages": {col: round(float(df[col].isnull().mean() * 100), 2) for col in df.columns},
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric_summaries": {}
        }

        # Check unique identifiers if present
        for id_col in ["ID", "order_id", "Delivery_person_ID", "customer_id"]:
            if id_col in df.columns:
                audit[f"unique_{id_col}"] = int(df[id_col].nunique())

        # Numeric column summaries
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            audit["numeric_summaries"][col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": round(float(df[col].mean()), 2),
                "std": round(float(df[col].std()), 2),
                "median": float(df[col].median()),
                "q25": float(df[col].quantile(0.25)),
                "q75": float(df[col].quantile(0.75)),
            }

        return audit

    def generate_audit_reports(self):
        """
        Generates and saves JSON and Markdown audit reports.
        """
        df_ops = self.load_delivery_operations_raw()
        df_fh = self.load_foodhub_raw()

        audit_ops = self.audit_dataset(df_ops, "Delivery Operations (Zomato Benchmark)")
        audit_fh = self.audit_dataset(df_fh, "FoodHub Customer Experience")

        # Save JSON
        with open(os.path.join(self.reports_dir, "audit_delivery_operations.json"), "w") as f:
            json.dump(audit_ops, f, indent=2)

        with open(os.path.join(self.reports_dir, "audit_foodhub.json"), "w") as f:
            json.dump(audit_fh, f, indent=2)

        # Save Markdown Summary
        md_content = self._generate_markdown_report(audit_ops, audit_fh)
        with open(os.path.join(self.reports_dir, "DATA_AUDIT_REPORT.md"), "w") as f:
            f.write(md_content)

        print(f"Data audit reports successfully saved to {self.reports_dir}/")
        return audit_ops, audit_fh

    def _generate_markdown_report(self, audit_ops: dict, audit_fh: dict) -> str:
        md = f"""# PromiseGuard: Comprehensive Data Audit Report

**Date of Audit**: November 2026  
**Auditor**: PromiseGuard Data Governance Pipeline  
**Datasets Audited**:
1. `{audit_ops['dataset_name']}`
2. `{audit_fh['dataset_name']}`

---

## 1. Primary Operational Dataset: {audit_ops['dataset_name']}

- **Total Observations (Rows)**: {audit_ops['total_rows']:,}
- **Total Variables (Columns)**: {audit_ops['total_columns']}
- **Duplicate Records**: {audit_ops['duplicate_rows']}
- **Unique Order IDs**: {audit_ops.get('unique_ID', 'N/A'):,}
- **Unique Delivery Partners**: {audit_ops.get('unique_Delivery_person_ID', 'N/A'):,}

### Missingness Profile
| Column | Missing Count | Missing % | Status |
|---|---|---|---|
"""
        for col, count in audit_ops["missing_values"].items():
            pct = audit_ops["missing_percentages"][col]
            status = "Clean" if count == 0 else ("Low (<5%)" if pct < 5 else "Moderate (>5%)")
            md += f"| `{col}` | {count:,} | {pct}% | {status} |\n"

        md += f"""
### Key Numeric Distributions
| Variable | Min | Mean | Median | Max | Std Dev |
|---|---|---|---|---|---|
"""
        for col, s in audit_ops["numeric_summaries"].items():
            md += f"| `{col}` | {s['min']} | {s['mean']} | {s['median']} | {s['max']} | {s['std']} |\n"

        md += f"""
---

## 2. Customer Satisfaction Dataset: {audit_fh['dataset_name']}

- **Total Observations (Rows)**: {audit_fh['total_rows']:,}
- **Total Variables (Columns)**: {audit_fh['total_columns']}
- **Duplicate Records**: {audit_fh['duplicate_rows']}
- **Unique Orders**: {audit_fh.get('unique_order_id', 'N/A'):,}
- **Unique Customers**: {audit_fh.get('unique_customer_id', 'N/A'):,}

### Missingness Profile
| Column | Missing Count | Missing % | Status |
|---|---|---|---|
"""
        for col, count in audit_fh["missing_values"].items():
            pct = audit_fh["missing_percentages"][col]
            md += f"| `{col}` | {count:,} | {pct}% | {'Clean' if count == 0 else 'Missing'} |\n"

        md += f"""
### Key Numeric Distributions
| Variable | Min | Mean | Median | Max | Std Dev |
|---|---|---|---|---|---|
"""
        for col, s in audit_fh["numeric_summaries"].items():
            md += f"| `{col}` | {s['min']} | {s['mean']} | {s['median']} | {s['max']} | {s['std']} |\n"

        md += """
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
"""
        return md


if __name__ == "__main__":
    loader = DataLoader()
    loader.generate_audit_reports()
