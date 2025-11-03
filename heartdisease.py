# Heart Disease Prediction using Python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
data = pd.read_csv("heart.csv")  # Ensure heart.csv is in same folder
print("✅ Dataset loaded successfully!")
print(data.head())

# Step 2: Split data into features and target
X = data.iloc[:, :-1]   # All columns except last
y = data.iloc[:, -1]    # Last column is target (0 = No disease, 1 = Disease)

# Step 3: Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale the data (improves model accuracy)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("✅ Model training completed!")

# Step 6: Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")

# Step 7: Predict heart disease for a new patient
print("\n--- Heart Disease Risk Prediction ---")
try:
    age = float(input("Enter Age: "))
    sex = int(input("Sex (1 = Male, 0 = Female): "))
    cp = int(input("Chest Pain Type (0=Typical, 1=Atypical, 2=Non-Anginal, 3=Asymptomatic): "))
    trestbps = float(input("Resting Blood Pressure (in mm Hg): "))
    chol = float(input("Cholesterol Level (mg/dl): "))
    fbs = int(input("Fasting Blood Sugar >120mg/dl (1=Yes, 0=No): "))
    restecg = int(input("Resting ECG Results (0=Normal, 1=ST-T Abnormality, 2=LV Hypertrophy): "))
    thalach = float(input("Max Heart Rate Achieved: "))
    exang = int(input("Exercise Induced Angina (1=Yes, 0=No): "))
    oldpeak = float(input("ST Depression (decimal, e.g. 0.0–6.0): "))
    slope = int(input("Slope of Peak Exercise ST Segment (0=Upsloping, 1=Flat, 2=Downsloping): "))
    ca = int(input("Number of Major Vessels (0–3): "))
    thal = int(input("Thalassemia Type (0=Normal, 1=Fixed Defect, 2=Reversible Defect): "))

    new_data = [[age, sex, cp, trestbps, chol, fbs, restecg,
                 thalach, exang, oldpeak, slope, ca, thal]]
    new_data = scaler.transform(new_data)

    prediction = model.predict(new_data)[0]
    probability = model.predict_proba(new_data)[0][1]  # Probability of heart disease

    print("\nResult:")
    if prediction == 1:
        print(f"⚠️ High Risk: The person may have heart disease.")
    else:
        print(f"✅ Low Risk: The person is likely healthy.")

    print(f"📊 Estimated Heart Disease Probability: {probability * 100:.2f}%")

except Exception as e:
    print("Input error. Please enter valid numeric values.")
