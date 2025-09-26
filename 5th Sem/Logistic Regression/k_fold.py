from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix 

df = pd.read_csv(r"C:\Users\sriha\OneDrive\Documents\GitHub\College\AILAB\5th Sem\Logistic Regression\dummy_dataset_5200.csv")

le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

X = df[["Age", "Gender", "AccountBalance"]]
y = df["ActiveCustomer"]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=1000)
scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")

print("K-Fold Accuracies:", scores)
print("Average Accuracy:", scores.mean())



model.fit(X, y)


y_pred_cv = cross_val_predict(model, X, y, cv=kf)

cm = confusion_matrix(y, y_pred_cv)
print("\nConfusion Matrix (from 5-Fold CV Predictions):")
print(cm)


sample_data = pd.DataFrame([[35, 1, 50000]], columns=["Age", "Gender", "AccountBalance"])
sample_prediction = model.predict(sample_data)
print(f"Sample Prediction (Age=35, Gender=1, Balance=50000): {sample_prediction[0]}")