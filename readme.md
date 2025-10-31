# 🕵️‍♀️ Fake Profile Detection Using Machine Learning

This project detects **fake social media profiles** using a Machine Learning model.  
It analyzes profile details like followers, following, account age, bio, and profile picture to predict whether a profile is **Real** or **Fake**.

---

## 📘 Project Overview
- Load dataset of social media profiles  
- Train a **Random Forest Classifier**  
- Test model accuracy and display performance graphs  
- Save trained model and scaler  
- Predict new profile status

---

## 📂 Dataset
The dataset is in `data/fake_profiles.csv` with the following columns:

| Column | Description |
|--------|-------------|
| username_length | Number of characters in the username |
| followers_count | Number of followers |
| following_count | Number of accounts the user follows |
| posts_count | Number of posts |
| account_age | Account age in days |
| bio_exists | 1 if bio exists, 0 if not |
| profile_pic | 1 if profile picture exists, 0 if not |
| is_fake | Target: 1 = Fake, 0 = Real |

---

## ⚙️ How It Works
1. Load dataset  
2. Scale features using **StandardScaler**  
3. Split dataset into train and test sets  
4. Train **Random Forest Classifier**  
5. Evaluate accuracy, confusion matrix, ROC and Precision-Recall curves  
6. Save model and scaler (`.pkl` files)

---

## 🧩 Installation
1. Clone this repository:
```bash
git clone https://github.com/tavvapallavi/Fake-Profile-Detection-ML.git
cd Fake-Profile-Detection-ML

##install all python dependencies as mentioned below##

pip install pandas numpy scikit-learn matplotlib seaborn joblib

##Train the Model##
python train_model.py

##Use the prediction script##
python predict.py
