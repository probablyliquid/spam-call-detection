from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your pre-trained model
model = load_model("spam_detector.h5")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    country_code = request.form["country_code"]
    spam_reports = float(request.form["spam_reports"])
    saved_number = float(request.form["saved_number"])

    # Preprocess the input data (convert to float, handle other preprocessing steps)
    user_data = np.array([[country_code, spam_reports, saved_number]])
    user_data = user_data.astype("float32")

    # Make predictions using your loaded model
    confidence = model.predict(user_data)

    # Decision logic
    prediction = (
        "This is likely spam." if confidence > 0.5 else "This is likely not spam."
    )

    # Pass the prediction to the template
    return render_template(
        "index.html", prediction=prediction, confidence=confidence[0][0]
    )
