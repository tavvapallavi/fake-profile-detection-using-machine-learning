import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

np.random.seed(42)

data = {
    "username_length": np.random.randint(5, 20, 500),
    "followers_count": np.random.randint(0, 10000, 500),
    "following_count": np.random.randint(0, 10000, 500),
    "posts_count": np.random.randint(0, 1000, 500),
    "account_age": np.random.randint(0, 3650, 500),
    "bio_exists": np.random.choice([0, 1], 500),
    "profile_pic": np.random.choice([0, 1], 500),
    "is_fake": np.random.choice([0, 1], 500, p=[0.7, 0.3])
}

df = pd.DataFrame(data)
df.to_csv("data/fake_profiles.csv", index=False)
print("✅ Dataset generated at: data/fake_profiles.csv")

