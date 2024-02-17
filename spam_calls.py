import tensorflow as tf
import pandas as pd
import numpy as np

# Dataset
dataset = pd.read_csv("spam_calls.csv")
dataset.head()

# Data Cleanup
print(f"Orginal dataset shape: {dataset.shape}")
dataset = dataset.drop(columns=["ID", "Phone_Number"])
print(f"New dataset shape: {dataset.shape}")
dataset.head()

# Variables: X
X = dataset.drop(columns=["Decision"])
print(f"X shape: {X.shape}")
X.head()

# Variables: y
y = dataset[["Decision"]]
print(f"y shape: {y.shape}")
y.head()

# Model Creation
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

X = np.array(X).astype("float32")
y = np.array(y).astype("float32")

model = Sequential(
    [
        Dense(8, input_shape=(X.shape[1],), activation="relu"),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

# Model training

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=1000)


# User Input and Check the Data
def input_user_data():
    country_code = input("Enter Country Code (Without +): ")
    spam_reports = float(input("Enter Number of Spam Reports: "))
    saved_number = float(input("Is this a Saved Number (0 or 1): "))

    user_data = np.array([[country_code, spam_reports, saved_number]])
    user_data = user_data.astype("float32")
    return user_data


def decisionMaker(decisionVar):
    if decisionVar > 0.5:
        print("This is likely spam.")
    else:
        print("This is likely not spam.")


# Decision
userData = input_user_data()
confidence = model.predict(userData)
decisionMaker(confidence)

# Model Prediction / Confidence
print(f"Probability of being spam: {confidence[0][0] * 100}%")

# Save the trained model to an HDF5 file
model.save("spam_detector.h5")
