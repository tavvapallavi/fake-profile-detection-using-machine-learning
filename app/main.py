from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("models/fake_profile_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[key]) for key in [
            "username_length", "followers_count", "following_count",
            "posts_count", "account_age", "bio_exists", "profile_pic"
        ]]
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]
        result = "Fake Profile" if prediction == 1 else "Real Profile"
        return render_template('index.html', prediction_text=f"Result: {result}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

