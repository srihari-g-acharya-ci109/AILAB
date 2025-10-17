# # Simple Bayesian (Naive Bayes) spam detection with asymmetric misclassification costs
# # Requires: pip install scikit-learn pandas

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score

# # ------------------ Load and prepare data ------------------
# data = pd.read_csv("email.csv")   # Must have columns: Category, Message
# data['Category'] = data['Category'].map({'ham': 0, 'spam': 1})

# X_train, X_test, y_train, y_test = train_test_split(
#     data['Message'], data['Category'], test_size=0.2, random_state=42
# )

# vectorizer = CountVectorizer()
# X_train_counts = vectorizer.fit_transform(X_train)
# X_test_counts = vectorizer.transform(X_test)

# # ------------------ Train Naive Bayes model ------------------
# model = MultinomialNB()
# model.fit(X_train_counts, y_train)

# y_pred = model.predict(X_test_counts)
# print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))

# # ------------------ Define asymmetric costs ------------------
# cost_spam_as_legit = 1   # Cost of misclassifying spam as legit
# cost_legit_as_spam = 5   # Cost of misclassifying legit as spam

# msg = ["Congratulations! You won a free ticket. Call now!"]
# msg_vec = vectorizer.transform(msg)

# probs = model.predict_proba(msg_vec)[0]
# P_legit_given_msg = probs[0]
# P_spam_given_msg = probs[1]

# risk_classify_spam = cost_legit_as_spam * P_legit_given_msg
# risk_classify_legit = cost_spam_as_legit * P_spam_given_msg

# print("\nP(Spam | message):", round(P_spam_given_msg, 3))
# print("P(Legit | message):", round(P_legit_given_msg, 3))
# print("Risk if classify as Spam:", round(risk_classify_spam, 3))
# print("Risk if classify as Legit:", round(risk_classify_legit, 3))

# if risk_classify_spam < risk_classify_legit:
#     print("\nDecision (Bayesian with Cost): Classify as SPAM")
# else:
#     print("\nDecision (Bayesian with Cost): Classify as LEGIT")



# Simple Bayesian (Naive Bayes) spam detection with asymmetric misclassification costs
# Requires: pip install scikit-learn pandas

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ------------------ Load and clean data ------------------
data = pd.read_csv("email.csv")   # File must have columns: Category, Message

# Standardize column names (in case CSV has extra spaces)
data.columns = data.columns.str.strip()

# Map categories safely
data['Category'] = data['Category'].str.strip().str.lower().map({'ham': 0, 'spam': 1})

# Drop any rows where mapping failed
data = data.dropna(subset=['Category', 'Message'])

# ------------------ Split and vectorize ------------------
X_train, X_test, y_train, y_test = train_test_split(
    data['Message'], data['Category'], test_size=0.2, random_state=42
)

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# ------------------ Train Naive Bayes model ------------------
model = MultinomialNB(alpha=0.1)
model.fit(X_train_counts, y_train)

y_pred = model.predict(X_test_counts)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))

# ------------------ Define asymmetric costs ------------------
cost_spam_as_legit = 3   # Cost of misclassifying spam as legit
cost_legit_as_spam = 5   # Cost of misclassifying legit as spam

# ------------------ Predict custom message ------------------
# msg = ["Check out this new offer for you"]
# msg = ["Exclusive offer just for you"]  
msg = ["Limited time deal waiting for you"]  
# msg = ["You might be eligible for a free gift"] 
# msg = ["Special offer just for you, claim your discount now"]
# msg = ["Special offer just for you, claim your discount now"]
# msg = ["You have an offer"]  # very neutral
# msg = ["Limited opportunity"]  # neutral wording
# msg = ["Don't miss out on this"]  # ambiguous, not “spammy” enough



msg_vec = vectorizer.transform(msg)


original_class = model.predict(msg_vec)[0]
if original_class == 1:
    print("\nOriginal Prediction (without considering costs): SPAM")
else:
    print("\nOriginal Prediction (without considering costs): LEGIT")


# Get posterior probabilities
probs = model.predict_proba(msg_vec)[0]
P_legit_given_msg = probs[0]
P_spam_given_msg = probs[1]

# Compute expected risks
risk_classify_spam = cost_legit_as_spam * P_legit_given_msg
risk_classify_legit = cost_spam_as_legit * P_spam_given_msg

print("\nP(Spam | message):", round(P_spam_given_msg, 3))
print("P(Legit | message):", round(P_legit_given_msg, 3))
print("Risk if classify as Spam:", round(risk_classify_spam, 3))
print("Risk if classify as Legit:", round(risk_classify_legit, 3))

if risk_classify_spam < risk_classify_legit:
    print("\nDecision (Bayesian with Cost): Classify as SPAM")
else:
    print("\nDecision (Bayesian with Cost): Classify as LEGIT")
