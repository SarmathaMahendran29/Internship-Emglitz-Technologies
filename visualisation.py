# visualization.py
# This file is for generating graphs for your report

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("heart.csv")

# 1️⃣ Target Distribution
data['target'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Heart Disease Distribution')
plt.xlabel('Target (0 = No Disease, 1 = Disease)')
plt.ylabel('Number of Patients')
plt.show()

# 2️⃣ Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), cmap='coolwarm', annot=False)
plt.title('Feature Correlation Heatmap')
plt.show()

# 3️⃣ Age vs Cholesterol (extra graph - optional)
plt.scatter(data['age'], data['chol'], color='blue', alpha=0.6)
plt.title('Age vs Cholesterol Level')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.show()

print("✅ All graphs generated successfully. Take screenshots for your report.")
