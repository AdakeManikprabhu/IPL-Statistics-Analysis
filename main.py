# ============================================
# 1. UPLOAD FILES IN COLAB
# ============================================
from google.colab import files
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql.functions import floor


print("Upload: IPL_BallByBall2008_2024.csv")
bb_upload = files.upload()

print("Upload: team_performance_dataset_2008to2024.csv")
team_upload = files.upload()

print("Upload: auction.csv")
auction_upload = files.upload()

ball_path = list(bb_upload.keys())[0]
team_path = list(team_upload.keys())[0]
auction_path = list(auction_upload.keys())[0]


# ============================================
# 2. CREATE OUTPUT FOLDERS
# ============================================
import os
from pathlib import Path

out_dir = Path("outputs")
charts_dir = out_dir / "charts"
tables_dir = out_dir / "tables"

out_dir.mkdir(exist_ok=True)
charts_dir.mkdir(exist_ok=True)
tables_dir.mkdir(exist_ok=True)


# ============================================
# 3. START SPARK SESSION (NO WINUTILS WARNING IN COLAB)
# ============================================
!pip install pyspark --quiet

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, expr, sum as spark_sum, count, avg, desc, lit, regexp_replace
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, IntegerType, StringType

spark = SparkSession.builder.appName("IPL_PySpark_Project").getOrCreate()


# ============================================
# 4. LOAD CSVs IN SPARK
# ============================================
bb_df = spark.read.csv(ball_path, header=True, inferSchema=True)
team_df = spark.read.csv(team_path, header=True, inferSchema=True)
auction_df = spark.read.csv(auction_path, header=True, inferSchema=True)


# ============================================
# 5. CLEAN / RENAME COLUMNS
# ============================================
bb_df = (
    bb_df
    .withColumnRenamed("Match id", "match_id")
    .withColumnRenamed("Innings No", "inning_no")
    .withColumnRenamed("Ball No", "ball_no")
    .withColumnRenamed("Striker", "batter")
    .withColumnRenamed("Non Striker", "non_striker")
    .withColumnRenamed("runs_scored", "runs_scored")
    .withColumnRenamed("extras", "extras")
    .withColumnRenamed("Player Out", "player_out")
    .withColumnRenamed("wicket_confirmation", "wicket_flag")
    .withColumnRenamed("wicket_type", "wicket_type")
    .withColumn("is_wicket", col("wicket_flag").cast("int"))
)
bb_df = bb_df.withColumn("over", floor((col("ball_no") - 1) / 6) + 1)
bb_df = bb_df.withColumn(
    "match_phase",
    when(col("over") <= 6, "Powerplay")
    .when((col("over") >= 7) & (col("over") <= 15), "Middle")
    .otherwise("Death")
)


team_df = (
    team_df
    .withColumnRenamed("Match_ID", "match_id")
    .withColumnRenamed("Toss_Winner", "toss_winner")
    .withColumnRenamed("Toss_Decision", "toss_decision")
    .withColumnRenamed("Match_Winner", "match_winner")
    .withColumnRenamed("Win_Type", "win_type")
    .withColumnRenamed("First_Innings_Score", "first_innings_score")
    .withColumnRenamed("Powerplay_Scores", "powerplay_runs")
    .withColumnRenamed("Middle_Overs_Scores", "middle_overs_runs")
    .withColumnRenamed("Death_Overs_Scores", "death_overs_runs")
)


auction_df = (
    auction_df
    .withColumnRenamed("Player", "player_name")
    .withColumnRenamed("Team", "auction_team")
    .withColumnRenamed("Base price", "base_price")
    .withColumnRenamed("Winning bid", "sold_price")
    .withColumnRenamed("Country", "auction_country")
)

auction_df = auction_df.withColumn(
    "sold_price_num",
    regexp_replace(col("sold_price"), "[^0-9.]", "").cast("double")
)

auction_df = auction_df.withColumn(
    "base_price_num",
    regexp_replace(col("base_price"), "[^0-9.]", "").cast("double")
)


# ============================================
# 6. BOWLER STATS
# ============================================
bowler_stats = (
    bb_df.groupBy("bowler")
    .agg(
        spark_sum(col("runs_scored") + col("extras")).alias("runs_conceded"),
        spark_sum("is_wicket").alias("wickets"),
        count("ball_no").alias("balls_bowled")
    )
    .withColumn("overs_bowled", col("balls_bowled") / 6.0)
    .withColumn("economy", col("runs_conceded") / col("overs_bowled"))
)

top_bowlers = bowler_stats.orderBy(desc("wickets")).limit(20)
top_bowlers_pd = top_bowlers.toPandas()


# ============================================
# 7. VISUALIZE – TOP BOWLERS
# ============================================
plt.figure(figsize=(10,6))
plot_df_b = top_bowlers_pd.nlargest(10, "wickets")[::-1]
plt.barh(plot_df_b["bowler"], plot_df_b["wickets"])
plt.xlabel("Wickets")
plt.title("Top 10 IPL Bowlers by Wickets (2008–2024)")
plt.tight_layout()
plt.savefig(charts_dir/"top10_bowlers.png", dpi=150)
plt.show()


# ============================================
# 8. BATSMEN STATS
# ============================================
batting_stats = (
    bb_df
    .groupBy("batter")
    .agg(
        spark_sum("runs_scored").alias("total_runs"),
        count("runs_scored").alias("balls_faced")
    )
)

top_batsmen = batting_stats.orderBy(desc("total_runs")).limit(20)
top_batsmen_pd = top_batsmen.toPandas()
top_batsmen_pd.to_csv(tables_dir/"top_batsmen.csv", index=False)


plt.figure(figsize=(10,6))
plot_df_t = top_batsmen_pd.nlargest(10, "total_runs")[::-1]
plt.barh(plot_df_t["batter"], plot_df_t["total_runs"])
plt.xlabel("Total Runs")
plt.title("Top 10 IPL Batsmen")
plt.tight_layout()
plt.savefig(charts_dir/"top10_batsmen.png", dpi=150)
plt.show()


# ============================================
# 9. TOP PARTNERSHIPS
# ============================================
from pyspark.sql.functions import udf
pair_udf = udf(lambda a, b: "::".join(sorted([str(a), str(b)])) if a and b else None, StringType())

partner_df = (
    bb_df.withColumn("pair", pair_udf(col("batter"), col("non_striker")))
    .groupBy("pair")
    .agg(
        spark_sum("runs_scored").alias("runs_as_pair"),
        count("pair").alias("balls_together")
    )
    .orderBy(desc("runs_as_pair"))
)

top_pairs_pd = partner_df.limit(30).toPandas()
top_pairs_pd.to_csv(tables_dir/"top_pairs.csv", index=False)


# VISUALIZE
plt.figure(figsize=(10,8))
pairs_plot = top_pairs_pd.nlargest(15, "runs_as_pair")[::-1]
plt.barh(pairs_plot["pair"], pairs_plot["runs_as_pair"])
plt.xlabel("Runs Together")
plt.title("Top 15 IPL Partnerships")
plt.tight_layout()
plt.savefig(charts_dir/"partnerships.png", dpi=150)
plt.show()


print("\n\n🎉 ALL PLOTS GENERATED SUCCESSFULLY!")
print("📁 Saved to: /content/outputs/charts and /content/outputs/tables")