# Databricks notebook source
import pandas as pd

silver_path = "/Workspace/MediaAnalytics/Silver_Layer/youtube_silver_cleaned.csv"
gold_folder = "/Workspace/MediaAnalytics/Gold_Layer"

df_silver = pd.read_csv(silver_path)

print("Silver Data Loaded:")
df_silver.head()

# COMMAND ----------

import numpy as np

df_gold = df_silver.copy()

# Engagement Score
df_gold["engagement_score"] = df_gold["likes"].fillna(0) + df_gold["replies"].fillna(0)

# User-level aggregates
user_stats = df_gold.groupby("username").agg({
    "comment_id": "count",
    "likes": "sum",
    "replies": "sum",
    "engagement_score": "sum"
}).reset_index()

user_stats.rename(columns={"comment_id": "total_comments"}, inplace=True)

user_stats.head()

     

# COMMAND ----------


gold_path = f"{gold_folder}/youtube_gold_aggregated.csv"

user_stats.to_csv(gold_path, index=False)

print("Gold File Saved Successfully At:", gold_path)


# COMMAND ----------


# DIM USER TABLE
dim_user = df_gold[["username", "user_channel"]].drop_duplicates()

dim_user["user_id"] = dim_user["username"].factorize()[0] + 1

dim_user_path = f"{gold_folder}/dim_user.csv"
dim_user.to_csv(dim_user_path, index=False)

print("Created:", dim_user_path)

# COMMAND ----------

dim_video = df_gold[["video_id", "url"]].drop_duplicates()

dim_video["video_sk"] = dim_video["video_id"].factorize()[0] + 1

dim_video_path = f"{gold_folder}/dim_video.csv"
dim_video.to_csv(dim_video_path, index=False)

print("Created:", dim_video_path)


# COMMAND ----------

df_gold["date"] = pd.to_datetime(df_gold["date"], errors="coerce")

dim_date = pd.DataFrame()
dim_date["date"] = df_gold["date"].dropna().unique()

dim_date["date"] = pd.to_datetime(dim_date["date"])
dim_date["date_sk"] = dim_date["date"].dt.strftime("%Y%m%d").astype(int)
dim_date["year"] = dim_date["date"].dt.year
dim_date["month"] = dim_date["date"].dt.month
dim_date["day"] = dim_date["date"].dt.day
dim_date["day_of_week"] = dim_date["date"].dt.day_name()

dim_date_path = f"{gold_folder}/dim_date.csv"
dim_date.to_csv(dim_date_path, index=False)

print("Created:", dim_date_path)

# COMMAND ----------

fact_comments = df_gold[[
    "comment_id", "comment_text", "likes", "replies",
    "username", "video_id", "date"
]].copy()

# Join surrogate keys
fact_comments = fact_comments.merge(dim_user, on="username", how="left")
fact_comments = fact_comments.merge(dim_video, on="video_id", how="left")
fact_comments = fact_comments.merge(dim_date, on="date", how="left")

fact_comments_path = f"{gold_folder}/fact_comments.csv"
fact_comments.to_csv(fact_comments_path, index=False)

print("Created:", fact_comments_path)

# If date is missing, generate from year, month, day
if "date" not in fact_comments.columns or fact_comments["date"].isna().all():

    print("⚠ Date is missing — generating from year, month, day...")

    fact_comments["date"] = pd.to_datetime(
        fact_comments[["year", "month", "day"]],
        errors="coerce"
    )

    # If any rows fail to create a date, fill with today's date
    fact_comments["date"] = fact_comments["date"].fillna(pd.Timestamp.today())

print("Generated Date Column:")
print(fact_comments[["year", "month", "day", "date"]].head())


     

# COMMAND ----------

fact_user_stats_path = f"{gold_folder}/fact_user_stats.csv"
user_stats.to_csv(fact_user_stats_path, index=False)

print("Created:", fact_user_stats_path)