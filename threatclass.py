import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Generate synthetic data
np.random.seed(0)  # For reproducibility
n_samples = 500

data = {
    "Speed": np.random.uniform(200, 1200, n_samples),  # Speed in km/h
    "Altitude": np.random.uniform(500, 10000, n_samples),  # Altitude in meters
    "RCS": np.random.uniform(0.1, 10, n_samples),  # Radar Cross-Section (RCS)
    "Distance": np.random.uniform(1, 50, n_samples),  # Distance in kilometers
    "Label": np.random.choice([0, 1], size=n_samples),  # Labels (0 for Friendly, 1 for Hostile)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Split the data into features and labels
X = df.drop("Label", axis=1)
y = df["Label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the SVM classifier
svm_classifier = SVC(kernel="linear", C=1.0)
svm_classifier.fit(X_train, y_train)

# Make predictions
y_pred = svm_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)
