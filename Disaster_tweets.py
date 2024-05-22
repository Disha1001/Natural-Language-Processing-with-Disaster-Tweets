import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
X_train, X_val, y_train, y_val = train_test_split(train_df["text"], train_df["target"], test_size=0.2, random_state=42)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)
X_test_tfidf = tfidf_vectorizer.transform(test_df["text"])
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
val_predictions = model.predict(X_val_tfidf)
f1 = f1_score(y_val, val_predictions)
print("F1 Score:", f1)
test_predictions = model.predict(X_test_tfidf)
submission_df = pd.DataFrame({"id": test_df["id"], "target": test_predictions})
submission_df.to_csv("submission.csv", index=False)
submission_df = pd.read_csv("submission.csv")
print(submission_df)
