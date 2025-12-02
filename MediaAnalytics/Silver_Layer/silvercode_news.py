# Databricks notebook source

import pandas as pd
import json

bronze_path = "/Workspace/MediaAnalytics/Bronze_Layer/news_raw.json"
silver_path = "/Workspace/MediaAnalytics/Silver_Layer/news_cleaned.csv"

# Load raw JSON
with open(bronze_path, "r") as f:
    data = json.load(f)

articles = data.get("articles", [])

df = pd.json_normalize(articles)

# ---------------------------
# Cleaning
# ---------------------------

df.drop_duplicates(subset=["url"], inplace=True)
df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
df["title"] = df["title"].str.strip()
df["description"] = df["description"].str.strip()
df["content"] = df["content"].str.strip()

# Remove rows with no title
df = df.dropna(subset=["title"])

df.to_csv(silver_path, index=False)

print("Silver Layer Saved:", silver_path)