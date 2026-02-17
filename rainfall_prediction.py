# ==========================================================
# PROJECT: Exploratory Analysis of Rainfall Data in India
# PURPOSE: Agricultural Insights and Rainfall Pattern Study
# DATASET: Rainfall 01-15.csv
# ==========================================================

# -------------------------------
# 1. Import Libraries
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# -------------------------------
# 2. Load Dataset
# -------------------------------
df = pd.read_csv("Rainfall 01-15.csv")

print("Dataset Loaded Successfully")

# -------------------------------
# 3. Data Understanding
# -------------------------------
print("\nShape:", df.shape)
print("\nColumns:\n", df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# -------------------------------
# 4. Check Missing Values
# -------------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing numeric values with median
df.fillna(df.median(numeric_only=True), inplace=True)

# -------------------------------
# 5. Data Visualization
# -------------------------------

# Annual rainfall distribution
plt.figure()
sns.histplot(df["ANNUAL"], bins=30, kde=True)
plt.title("Distribution of Annual Rainfall")
plt.show()

# -------------------------------
# 6. Annual Rainfall Trend
# -------------------------------
annual_trend = df.groupby("YEAR")["ANNUAL"].mean()

plt.figure()
annual_trend.plot()
plt.title("Annual Rainfall Trend in India")
plt.xlabel("Year")
plt.ylabel("Rainfall (mm)")
plt.show()

# -------------------------------
# 7. Monsoon Analysis (Agriculture Critical)
# -------------------------------
monsoon_trend = df.groupby("YEAR")["Jun-Sep"].mean()

plt.figure()
monsoon_trend.plot()
plt.title("Monsoon Rainfall Trend")
plt.xlabel("Year")
plt.ylabel("Rainfall (mm)")
plt.show()

# -------------------------------
# 8. Seasonal Contribution
# -------------------------------
season_cols = ["Jan-Feb", "Mar-May", "Jun-Sep", "Oct-Dec"]

season_avg = df[season_cols].mean()

plt.figure()
season_avg.plot(kind="bar")
plt.title("Seasonal Rainfall Contribution")
plt.ylabel("Average Rainfall")
plt.show()

# -------------------------------
# 9. Monthly Rainfall Pattern
# -------------------------------
monthly_cols = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]

monthly_avg = df[monthly_cols].mean()

plt.figure()
monthly_avg.plot(marker="o")
plt.title("Average Monthly Rainfall Pattern")
plt.ylabel("Rainfall")
plt.show()

# -------------------------------
# 10. Subdivision Analysis
# -------------------------------
subdivision_avg = df.groupby("SUBDIVISION")["ANNUAL"].mean().sort_values(ascending=False)

plt.figure(figsize=(10,8))
subdivision_avg.head(15).plot(kind="barh")
plt.title("Top 15 Rainfall Subdivisions")
plt.xlabel("Rainfall")
plt.show()

# -------------------------------
# 11. Correlation Heatmap
# -------------------------------
plt.figure(figsize=(10,8))
sns.heatmap(df.select_dtypes(include=np.number).corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# 12. Agriculture Risk Analysis
# -------------------------------

# Weak monsoon years
weak_monsoon = monsoon_trend[monsoon_trend < monsoon_trend.mean()]

print("\nYears with Weak Monsoon (Drought Risk):")
print(list(weak_monsoon.index))

# High rainfall years
high_rainfall = annual_trend[annual_trend > annual_trend.mean()]

print("\nHigh Rainfall Years (Flood Risk):")
print(list(high_rainfall.index))

# -------------------------------
# 13. Monsoon Dependency Ratio
# -------------------------------
df["Monsoon_Ratio"] = df["Jun-Sep"] / df["ANNUAL"]

plt.figure()
sns.histplot(df["Monsoon_Ratio"], bins=30)
plt.title("Monsoon Dependency Distribution")
plt.show()

print("\nAverage Monsoon Dependency:", df["Monsoon_Ratio"].mean())

# -------------------------------
# 14. Key Insights
# -------------------------------
print("\n===== Key Insights =====")
print("Average Annual Rainfall:", df["ANNUAL"].mean())
print("Highest Rainfall Subdivision:", subdivision_avg.idxmax())

print("\nEDA Completed Successfully!")
