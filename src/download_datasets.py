import os
import requests
import pandas as pd

def download_file(urls, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"File already exists: {dest_path} ({os.path.getsize(dest_path)} bytes)")
        return True
    
    for url in urls:
        print(f"Trying to download from {url}...")
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 1000:
                with open(dest_path, "wb") as f:
                    f.write(r.content)
                print(f"Successfully downloaded {dest_path} ({len(r.content)} bytes) from {url}")
                return True
            else:
                print(f"Failed with status code {r.status_code}")
        except Exception as e:
            print(f"Error fetching {url}: {e}")
    return False

if __name__ == "__main__":
    # 1. FoodHub Order dataset
    foodhub_urls = [
        "https://raw.githubusercontent.com/manjugovindarajan/FoodHub-Order-Analysis-using-Python/main/foodhub_order.csv",
        "https://raw.githubusercontent.com/prneidhardt/Python-Foundations/main/foodhub_order.csv",
        "https://raw.githubusercontent.com/briananotado/FoodHub/main/foodhub_order.csv"
    ]
    foodhub_dest = "data/raw/foodhub_order.csv"
    download_file(foodhub_urls, foodhub_dest)

    # 2. Indian Food Delivery Operations dataset
    zomato_urls = [
        "https://raw.githubusercontent.com/T1burski/delivery_time_prediction/main/train.csv",
        "https://raw.githubusercontent.com/MisterAare/delivery_time_prediction/main/train.csv",
        "https://raw.githubusercontent.com/debaratibose/Delivery-Time-Prediction/master/train.csv",
        "https://raw.githubusercontent.com/gauravkolhe44/Zomato-Delivery-Time-Prediction/main/deliverytime.csv"
    ]
    zomato_dest = "data/raw/delivery_operations.csv"
    download_file(zomato_urls, zomato_dest)
