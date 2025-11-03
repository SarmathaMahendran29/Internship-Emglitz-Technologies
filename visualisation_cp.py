# Chest Pain Type vs Heart Disease Visualization

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (make sure heart.csv is in the same folder)
data = pd.read_csv("heart.csv")

# Create the bar chart
plt.figure(figsize=(7,5))
sns.countplot(x='cp', hue='target', data=data)
plt.title("Chest Pain Type vs Heart Disease")
plt.xlabel("Chest Pain Type (0=Typical, 1=Atypical, 2=Non-Anginal, 3=Asymptomatic)")
plt.ylabel("Number of Patients")
plt.legend(title="Heart Disease", labels=["No Disease", "Disease"])
plt.tight_layout()

# Show and save
plt.show()
plt.savefig("chest_pain_vs_heart_disease.png")
