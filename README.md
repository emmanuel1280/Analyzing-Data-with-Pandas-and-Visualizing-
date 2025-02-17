# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target  # Add target column (species)

# Map target values to species names
iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Display the first few rows
print("First 5 rows of the dataset:")
print(iris_df.head())

# Explore the structure of the dataset
print("\nDataset information:")
print(iris_df.info())

# Check for missing values
print("\nMissing values in the dataset:")
print(iris_df.isnull().sum())

# Clean the dataset (no missing values in Iris dataset, but this is a general step)
iris_df.dropna(inplace=True)  # Drop rows with missing values (if any)

# Task 2: Basic Data Analysis
# Compute basic statistics
print("\nBasic statistics of numerical columns:")
print(iris_df.describe())

# Group by species and compute mean of numerical columns
print("\nMean of numerical columns grouped by species:")
print(iris_df.groupby('species').mean())

# Task 3: Data Visualization
# Set seaborn style for better-looking plots
sns.set(style="whitegrid")

# 1. Line chart (not applicable for Iris dataset, so we'll skip this)
# 2. Bar chart: Average sepal length per species
plt.figure(figsize=(8, 5))
sns.barplot(x='species', y='sepal length (cm)', data=iris_df, estimator=np.mean, ci=None)
plt.title("Average Sepal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Sepal Length (cm)")
plt.show()

# 3. Histogram: Distribution of petal length
plt.figure(figsize=(8, 5))
sns.histplot(iris_df['petal length (cm)'], bins=20, kde=True, color='blue')
plt.title("Distribution of Petal Length")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot: Sepal length vs. petal length
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=iris_df, palette='viridis')
plt.title("Sepal Length vs. Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

# Additional Customization: Pairplot to visualize relationships between all numerical columns
plt.figure(figsize=(10, 8))
sns.pairplot(iris_df, hue='species', palette='viridis')
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()

# Findings and Observations
print("\nFindings and Observations:")
print("1. Setosa species has the smallest sepal and petal lengths.")
print("2. Virginica species has the largest sepal and petal lengths.")
print("3. There is a clear distinction between species in the scatter plot of sepal length vs. petal length.")
print("4. The distribution of petal length is bimodal, indicating two distinct groups.")

