import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve)
import joblib, os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ----------------------------
# 1️⃣ Load dataset
# ----------------------------
df = pd.read_csv("data/fake_profiles.csv")

# ----------------------------
# 2️⃣ Features and target
# ----------------------------
X = df.drop("is_fake", axis=1)
y = df["is_fake"]

# ----------------------------
# 3️⃣ Scale features
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 4️⃣ Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ----------------------------
# 5️⃣ Train Random Forest
# ----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# 6️⃣ Predictions
# ----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]  # Probabilities for ROC / PR

# ----------------------------
# 7️⃣ Metrics
# ----------------------------
acc = accuracy_score(y_test, y_pred)
print("✅ Accuracy:", acc)
print("✅ Classification Report:\n", classification_report(y_test, y_pred, target_names=["Real","Fake"]))

# ----------------------------
# 8️⃣ Save model & scaler
# ----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fake_profile_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("✅ Model and scaler saved successfully!")

# ----------------------------
# 9️⃣ Confusion Matrix
# ----------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real','Fake'], yticklabels=['Real','Fake'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ----------------------------
# 🔟 Feature Importance
# ----------------------------
importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=features, palette='viridis')
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# ----------------------------
# 1️⃣1️⃣ ROC Curve
# ----------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()

# ----------------------------
# 1️⃣2️⃣ Precision-Recall Curve
# ----------------------------
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(recall, precision, color='purple', lw=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

# ----------------------------
# 1️⃣3️⃣ Feature Distribution (optional)
# ----------------------------
plt.figure(figsize=(12,8))
for i, col in enumerate(X.columns):
    plt.subplot(3,3,i+1)
    sns.histplot(df[col], bins=20, kde=True, color='cyan')
    plt.title(col)
plt.tight_layout()
plt.show()
