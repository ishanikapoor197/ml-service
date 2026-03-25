# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import joblib
# import os
# import json

# # ── Load Dataset ──────────────────────────────────────────────
# df = pd.read_csv("data/job_skills_dataset.csv")

# # Expand dataset by creating augmented user profiles
# # For each role, we simulate users who have SOME of the required skills
# augmented_rows = []
# np.random.seed(42)
# for _, row in df.iterrows():
#     role = row["job_role"]
#     all_skills = [s.strip() for s in row["required_skills"].split(",")]
    
#     # Simulate 50 users per role with 40-80% of skills known
#     np.random.seed(42)
#     for _ in range(200):
#         n_known = np.random.randint(int(len(all_skills) * 0.3), int(len(all_skills) * 0.85))
#         known_skills = np.random.choice(all_skills, size=n_known, replace=False).tolist()
        

# # 🔥 ADD HERE
# extra_skills_pool = ["AWS", "GraphQL", "Redis", "CI/CD", "Testing", "Microservices"]

# if np.random.rand() > 0.7:
#     noise = np.random.choice(extra_skills_pool)
#     if noise not in known_skills:
#         known_skills.append(noise)
#         augmented_rows.append({
#             "target_role": role,
#             "user_skills": ", ".join(known_skills),
#             "all_required_skills": row["required_skills"]
#         })

# aug_df = pd.DataFrame(augmented_rows)

# # ── Feature Engineering ───────────────────────────────────────
# # Input: user's current skills (text)
# # Output: target role prediction

# X = aug_df["user_skills"].values
# y = aug_df["target_role"].values

# # TF-IDF on skill text
# def skill_tokenizer(text):
#     return [s.strip().lower() for s in text.split(",")]
# # vectorizer = TfidfVectorizer(
# #     tokenizer=lambda x: [s.strip().lower() for s in x.split(",")],
# vectorizer = TfidfVectorizer(
#     tokenizer=skill_tokenizer,
#     token_pattern=None,
#     ngram_range=(1, 1),
#     max_features=500
# )

# X_vec = vectorizer.fit_transform(X)

# # Train/Test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_vec, y, test_size=0.2, random_state=42
# )

# # ── Train Model ───────────────────────────────────────────────
# model = LogisticRegression(
#     max_iter=1000,
#     C=1.0,
#     multi_class='multinomial',
#     solver='lbfgs'
# )
# model.fit(X_train, y_train)

# # Evaluate
# y_pred = model.predict(X_test)
# acc = accuracy_score(y_test, y_pred)
# print(f"✅ Model Accuracy: {acc * 100:.2f}%")

# # ── Save Skills Dictionary ────────────────────────────────────
# skills_dict = {}
# for _, row in df.iterrows():
#     skills_dict[row["job_role"]] = [s.strip() for s in row["required_skills"].split(",")]

# # ── Save Everything ───────────────────────────────────────────
# os.makedirs("model", exist_ok=True)

# joblib.dump(model, "model/skill_model.pkl")
# joblib.dump(vectorizer, "model/vectorizer.pkl")

# with open("model/skills_dict.json", "w") as f:
#     json.dump(skills_dict, f, indent=2)

# print("✅ Model saved to model/")
# print(f"✅ Roles supported: {list(skills_dict.keys())}")
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import joblib
# import os
# import json

# # ── Load Dataset ──────────────────────────────────────────────
# df = pd.read_csv("data/job_skills_dataset.csv")

# # ── Set Random Seed (ONLY ONCE) ───────────────────────────────
# np.random.seed(42)

# # ── Expand Dataset (Generate ~5000 rows) ──────────────────────
# augmented_rows = []

# extra_skills_pool = [
#     "AWS", "GraphQL", "Redis", "CI/CD", "Testing",
#     "Microservices", "Docker", "Kubernetes"
# ]

# for _, row in df.iterrows():
#     role = row["job_role"]
#     all_skills = [s.strip() for s in row["required_skills"].split(",")]

#     for _ in range(200):  # 25 roles × 200 = ~5000 samples
#         n_known = np.random.randint(
#             int(len(all_skills) * 0.3),
#             int(len(all_skills) * 0.85)
#         )

#         known_skills = np.random.choice(
#             all_skills,
#             size=n_known,
#             replace=False
#         ).tolist()

#         # 🔥 Add noise (real-world simulation)
#         if np.random.rand() > 0.7:
#             noise = np.random.choice(extra_skills_pool)
#             if noise not in known_skills:
#                 known_skills.append(noise)

#         augmented_rows.append({
#             "target_role": role,
#             "user_skills": ", ".join(known_skills),
#             "all_required_skills": row["required_skills"]
#         })

# aug_df = pd.DataFrame(augmented_rows)

# print(f"✅ Generated dataset size: {len(aug_df)}")

# # ── Feature Engineering ───────────────────────────────────────
# X = aug_df["user_skills"].values
# y = aug_df["target_role"].values

# # Tokenizer (IMPORTANT for pickle)
# def skill_tokenizer(text):
#     return [s.strip().lower() for s in text.split(",")]

# # TF-IDF Vectorizer
# vectorizer = TfidfVectorizer(
#     tokenizer=skill_tokenizer,
#     token_pattern=None,
#     ngram_range=(1, 2),     # 🔥 better context
#     max_features=1000       # 🔥 more features
# )

# X_vec = vectorizer.fit_transform(X)

# # ── Train/Test Split ──────────────────────────────────────────
# X_train, X_test, y_train, y_test = train_test_split(
#     X_vec, y, test_size=0.2, random_state=42
# )

# # ── Train Model ───────────────────────────────────────────────
# model = LogisticRegression(
#     max_iter=1000,
#     C=1.0,
#     multi_class='multinomial',
#     solver='lbfgs',
#     class_weight='balanced'   # 🔥 handles imbalance
# )

# model.fit(X_train, y_train)

# # ── Evaluate ─────────────────────────────────────────────────
# y_pred = model.predict(X_test)
# acc = accuracy_score(y_test, y_pred)

# print(f"✅ Model Accuracy: {acc * 100:.2f}%")

# # 🔍 Check roles learned
# print("✅ Roles learned:", model.classes_)

# # ── Save Skills Dictionary ───────────────────────────────────
# skills_dict = {}

# for _, row in df.iterrows():
#     skills_dict[row["job_role"]] = [
#         s.strip() for s in row["required_skills"].split(",")
#     ]

# # ── Save Model + Vectorizer ──────────────────────────────────
# os.makedirs("model", exist_ok=True)

# joblib.dump(model, "model/skill_model.pkl")
# joblib.dump(vectorizer, "model/vectorizer.pkl")

# with open("model/skills_dict.json", "w") as f:
#     json.dump(skills_dict, f, indent=2)

# print("✅ Model saved to model/")
# print("✅ Ready for Flask API 🚀")
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import json

# ── Load Base Dataset ─────────────────────────────────────────
df = pd.read_csv("data/job_skills_dataset.csv")

# ── Set Random Seed ───────────────────────────────────────────
np.random.seed(42)

# ── Extra Feature Pools ───────────────────────────────────────
education_levels = ["B.Tech", "MSc", "MBA", "Self-taught"]
experience_levels = ["Entry", "Mid", "Senior"]
tools_pool = ["VS Code", "Docker", "Jupyter", "Figma", "JIRA", "GitHub"]
certifications_pool = ["AWS", "Azure", "Google", "Scrum", "None"]

extra_skills_pool = [
    "AWS", "GraphQL", "Redis", "CI/CD", "Testing",
    "Microservices", "Kubernetes"
]

# ── Generate Augmented Dataset (~5000 rows) ───────────────────
augmented_rows = []

for _, row in df.iterrows():
    role = row["job_role"]
    all_skills = [s.strip() for s in row["required_skills"].split(",")]

    for _ in range(200):  # 25 roles × 200 ≈ 5000 rows
        n_known = np.random.randint(
            int(len(all_skills) * 0.3),
            int(len(all_skills) * 0.85)
        )

        known_skills = np.random.choice(
            all_skills,
            size=n_known,
            replace=False
        ).tolist()

        # 🔥 Add noise (real-world simulation)
        if np.random.rand() > 0.7:
            noise = np.random.choice(extra_skills_pool)
            if noise not in known_skills:
                known_skills.append(noise)

        augmented_rows.append({
            "target_role": role,
            "user_skills": ", ".join(known_skills),
            "education": np.random.choice(education_levels),
            "experience_level": np.random.choice(experience_levels),
            "tools": np.random.choice(tools_pool),
            "certifications": np.random.choice(certifications_pool),
            "all_required_skills": row["required_skills"]
        })

aug_df = pd.DataFrame(augmented_rows)

print(f"✅ Generated dataset size: {len(aug_df)}")

# ── Feature Engineering ───────────────────────────────────────
# Combine all features into ONE text input
X = (
    aug_df["user_skills"] + " " +
    aug_df["education"] + " " +
    aug_df["experience_level"] + " " +
    aug_df["tools"] + " " +
    aug_df["certifications"]
).values

y = aug_df["target_role"].values

# ── Tokenizer (IMPORTANT for pickle) ─────────────────────────
def skill_tokenizer(text):
    return [s.strip().lower() for s in text.split(",")]

# ── TF-IDF Vectorizer ────────────────────────────────────────
vectorizer = TfidfVectorizer(
    tokenizer=skill_tokenizer,
    token_pattern=None,
    ngram_range=(1, 2),
    max_features=1000
)

X_vec = vectorizer.fit_transform(X)

# ── Train/Test Split ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# ── Train Model ──────────────────────────────────────────────
model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    multi_class='multinomial',
    solver='lbfgs',
    class_weight='balanced'
)

model.fit(X_train, y_train)

# ── Evaluate ────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Model Accuracy: {acc * 100:.2f}%")

# Check roles learned
print("✅ Roles learned:", model.classes_)

# ── Save Skills Dictionary ──────────────────────────────────
skills_dict = {}

for _, row in df.iterrows():
    skills_dict[row["job_role"]] = [
        s.strip() for s in row["required_skills"].split(",")
    ]

# ── Save Model + Vectorizer ─────────────────────────────────
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/skill_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

with open("model/skills_dict.json", "w") as f:
    json.dump(skills_dict, f, indent=2)

print("✅ Model saved to model/")
print("🚀 Ready for Flask API")