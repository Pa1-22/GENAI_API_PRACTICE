# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# COMMAND ----------

bronze_path = "/Workspace/MediaAnalytics/Bronze_Layer/Youtube-dataset-sample.csv"

# STEP 1: LOAD RAW BRONZE DATA

df = pd.read_csv(bronze_path)
print("Original Bronze shape:", df.shape)


# STEP 2: BASIC RAW CLEANING (Bronze Rules)

# Remove duplicates
df = df.drop_duplicates()

# Remove leading / trailing spaces (Fix: applymap deprecated → use map)
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()

# Replace blank values with NaN
df = df.replace(r'^\s*$', pd.NA, regex=True)

# ---- FIXED CRITICAL COLUMNS ----
critical_cols = ["comment_id", "comment_text", "username", "video_id"]

df = df.dropna(subset=critical_cols)

# Reset index
df.reset_index(drop=True, inplace=True)

print("Cleaned Bronze shape:", df.shape)



# STEP 3: OVERWRITE BRONZE RAW FILE
df.to_csv("/Workspace/MediaAnalytics/Bronze_Layer/Youtube-cleaned.csv", 
          index=False)