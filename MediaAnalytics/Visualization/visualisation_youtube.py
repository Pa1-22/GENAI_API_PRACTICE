# Databricks notebook source
import pandas as pd


viz_folder = "/Workspace/MediaAnalytics/Visualization"
gold_folder = "/Workspace/MediaAnalytics/Gold_Layer"

fact_comments = pd.read_csv(f"{gold_folder}/fact_comments.csv")
fact_user_stats = pd.read_csv(f"{gold_folder}/fact_user_stats.csv")
dim_user = pd.read_csv(f"{gold_folder}/dim_user.csv")
dim_video = pd.read_csv(f"{gold_folder}/dim_video.csv")
dim_date = pd.read_csv(f"{gold_folder}/dim_date.csv")

print("Data Loaded Successfully!")


# COMMAND ----------


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# COMMAND ----------

import os

os.makedirs(viz_folder, exist_ok=True)

top_users = fact_user_stats.sort_values(
    "engagement_score",
    ascending=False
).head(10)

plt.figure(figsize=(10, 6))
plt.bar(
    top_users["username"],
    top_users["engagement_score"]
)
plt.xticks(rotation=45)
plt.title("Top 10 Users by Engagement Score")
plt.xlabel("Username")
plt.ylabel("Engagement Score")

path = f"{viz_folder}/top_users.png"
plt.savefig(path, bbox_inches="tight")
plt.show()

print("Saved:", path)

# COMMAND ----------

import numpy as np
import pandas as pd

# Random dates within 6 months
fact_comments["date"] = pd.to_datetime(
    np.random.choice(
        pd.date_range("2024-01-01", "2024-06-30"),
        size=len(fact_comments)
    )
)



# COMMAND ----------

df_time = fact_comments.copy()

comments_over_time = (
    df_time.groupby("date")["comment_id"]
    .count()
    .reset_index()
    .sort_values("date")
)

plt.figure(figsize=(7,7))
plt.plot(comments_over_time["date"], comments_over_time["comment_id"])
plt.xticks(rotation=45)
plt.title("Comments Over Time")
plt.xlabel("Date")
plt.ylabel("Total Comments")
path = f"{viz_folder}/comments_over_time.png"
plt.savefig(path, bbox_inches="tight")
plt.show()



# COMMAND ----------


pie_users = fact_user_stats.sort_values("total_comments", ascending=False).head(5)

plt.figure(figsize=(8,8))
plt.pie(pie_users["total_comments"], labels=pie_users["username"], autopct='%1.1f%%')
plt.title("Top 5 Users by Comment Share")

path = f"{viz_folder}/pie_top_users.png"
plt.savefig(path, bbox_inches="tight")
plt.show()

print("Saved:", path)

# COMMAND ----------

channel_activity = fact_comments.groupby("user_channel")["comment_id"] \
                                .count() \
                                .sort_values(ascending=False) \
                                .head(10)

plt.figure(figsize=(12,6))
plt.bar(channel_activity.index, channel_activity.values)
plt.xticks(rotation=45)
plt.title("📺 Most Active Channels")
plt.xlabel("Channel")
plt.ylabel("Comments")

path = f"{viz_folder}/active_channels.png"
plt.savefig(path, bbox_inches="tight")
plt.show()

print("Saved:", path)

     

# COMMAND ----------

import seaborn as sns

plt.figure(figsize=(8,6))
sns.heatmap(fact_comments[["likes","replies"]].corr(), annot=True, cmap="coolwarm")

plt.title("🔥 Correlation Heatmap - Likes vs Replies")

path = f"{viz_folder}/engagement_heatmap.png"
plt.savefig(path, bbox_inches="tight")
plt.show()

print("Saved:", path)

# COMMAND ----------

top_users = fact_user_stats.sort_values("engagement_score", ascending=False).head(10)

plt.figure(figsize=(12,6))
plt.bar(top_users["username"], top_users["likes"], label="Likes")
plt.bar(top_users["username"], top_users["replies"], bottom=top_users["likes"], label="Replies")

plt.xticks(rotation=45)
plt.title("Likes vs Replies Contribution to Engagement")
plt.xlabel("Username")
plt.ylabel("Count")
plt.legend()
plt.show()
