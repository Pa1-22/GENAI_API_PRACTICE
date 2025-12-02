# Databricks notebook source
import requests
import pandas as pd
import json

# ------------------------------------------
# GNEWS API
# ------------------------------------------
API_KEY = "c99aca8e684d2bcdf3f79b5338619c74"
query = "technology"
url = f"https://gnews.io/api/v4/search?q={query}&token={API_KEY}"

response = requests.get(url)
data = response.json()

# ------------------------------------------
# Save RAW Bronze File
# ------------------------------------------
bronze_path = "/Workspace/MediaAnalytics/Bronze_Layer/news_raw.json"

with open(bronze_path, "w") as f:
    json.dump(data, f, indent=4)

print("Bronze layer raw file saved:", bronze_path)