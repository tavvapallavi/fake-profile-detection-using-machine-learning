import joblib
import numpy as np

# Load model and scaler
model = joblib.load("models/fake_profile_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Example profile: [username_length, followers_count, following_count, posts_count, account_age, bio_exists, profile_pic]
sample_profile = [10, 500, 300, 50, 800, 1, 1]

# Scale
sample_scaled = scaler.transform([sample_profile])

# Predict
prediction = model.predict(sample_scaled)[0]

# Output
result = "Fake Profile" if prediction == 1 else "Real Profile"
print("Prediction:", result)

