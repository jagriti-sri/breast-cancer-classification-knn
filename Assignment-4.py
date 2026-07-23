
# Topic: Breast Cancer Classification using K-Nearest Neighbors (KNN)


# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# Task 1: Data Understanding


# Load the dataset
df = pd.read_csv("data.csv")

print(" FIRST FIVE RECORDS")
print(df.head())

print("\n DATASET INFORMATION")
print(df.info())

print("\n SUMMARY STATISTICS")
print(df.describe())

# Identify numerical features
numerical_features = df.select_dtypes(include=["int64", "float64"]).columns

print("\n NUMERICAL FEATURES")
print(list(numerical_features))

# Target variable
target_variable = "diagnosis"

print("\n TARGET VARIABLE")
print(target_variable)

# Task 2: Data Preprocessing

print("\n MISSING VALUES")
print(df.isnull().sum())

# Remove unnecessary columns
if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)

if "Unnamed: 32" in df.columns:
    df.drop("Unnamed: 32", axis=1, inplace=True)

# Encode target variable
encoder = LabelEncoder()
df["diagnosis"] = encoder.fit_transform(df["diagnosis"])

print("\nTarget variable encoded successfully.")

# Features and Target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples :", len(X_test))


# Task 3: Model Development


# KNN Classifier with K = 5
model = KNeighborsClassifier(n_neighbors=5)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("\nModel trained successfully.")

# Task 4: Model Evaluation


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n MODEL EVALUATION")

print("Accuracy Score :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1 Score :", f1)

print("\nConfusion Matrix")
print(cm)

