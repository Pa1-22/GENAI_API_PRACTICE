# Databricks notebook source
import pandas as pd

bronze_path = "/Workspace/MediaAnalytics/Bronze_Layer/Youtube-cleaned.csv"

# Load bronze cleaned data
df_bronze = pd.read_csv(bronze_path)

print("Bronze cleaned data loaded!")
df_bronze.head()


# COMMAND ----------

# ---- SILVER LAYER TRANSFORMATIONS ----

df_silver = df_bronze.copy()

# Convert date column to datetime
df_silver["date"] = pd.to_datetime(df_silver["date"], errors="coerce")

# Convert numeric columns properly
numeric_cols = ["likes", "replies", "replies_value"]
for col in numeric_cols:
    df_silver[col] = pd.to_numeric(df_silver[col], errors="coerce")

# Drop rows where required fields are missing
required_cols = ["comment_id", "comment_text", "username", "video_id"]
df_silver = df_silver.dropna(subset=required_cols)

# Add derived column: comment length
df_silver["comment_length"] = df_silver["comment_text"].astype(str).apply(len)

# Reset index
df_silver.reset_index(drop=True, inplace=True)

print("Silver ETL completed!")
df_silver.head()

# COMMAND ----------

silver_output_path = "/Workspace/MediaAnalytics/Silver_Layer/youtube_silver_cleaned.csv"

df_silver.to_csv(silver_output_path, index=False)

print("Silver file created:", silver_output_path)

     