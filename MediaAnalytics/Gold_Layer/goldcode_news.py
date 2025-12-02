# Databricks notebook source
import pandas as pd

silver_path = "/Workspace/MediaAnalytics/Silver_Layer/news_cleaned.csv"
gold_folder = "/Workspace/MediaAnalytics/Gold_Layer"

df = pd.read_csv(silver_path)

# 1) dim_article
dim_article = df[["url", "title", "description", "content"]].drop_duplicates()
dim_article["article_id"] = dim_article["url"].astype("category").cat.codes

# 2) dim_source
dim_source = df[["source.name"]].drop_duplicates()
dim_source["source_id"] = dim_source["source.name"].astype("category").cat.codes

# 3) dim_date
df["publishedAt"] = pd.to_datetime(df["publishedAt"])
dim_date = pd.DataFrame()
dim_date["date"] = df["publishedAt"].dt.date.unique()
dim_date["date"] = pd.to_datetime(dim_date["date"])
dim_date["date_id"] = dim_date["date"].dt.strftime("%Y%m%d")

# Save
dim_article.to_csv(f"{gold_folder}/dim_article.csv", index=False)
dim_source.to_csv(f"{gold_folder}/dim_source.csv", index=False)
dim_date.to_csv(f"{gold_folder}/dim_date.csv", index=False)


# COMMAND ----------

fact_news = df.copy()

fact_news["article_id"] = fact_news["url"].astype("category").cat.codes
fact_news["source_id"] = fact_news["source.name"].astype("category").cat.codes
fact_news["date_id"] = pd.to_datetime(fact_news["publishedAt"]).dt.strftime("%Y%m%d")

fact_news = fact_news[["article_id", "source_id", "date_id", "title", "url"]]

fact_news.to_csv(f"{gold_folder}/fact_news.csv", index=False)

print("Gold Fact & Dimension tables created successfully!")
