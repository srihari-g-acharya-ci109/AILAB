import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv(r"C:\Users\sriha\OneDrive\Documents\GitHub\College\AILAB\5th Sem\Logistic Regression\dummy_dataset_5200.csv")

le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])  

X = df[["Age", "Gender", "AccountBalance"]]
y = df["ActiveCustomer"]


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

val_acc = accuracy_score(y_val, model.predict(X_val))
test_acc = accuracy_score(y_test, model.predict(X_test))

print("Train size:", X_train.shape, "Validation size:", X_val.shape, "Test size:", X_test.shape)
print("Validation Accuracy:", val_acc)
print("Test Accuracy:", test_acc)


cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)


sample_data = pd.DataFrame([[35, 1, 50000]], columns=["Age", "Gender", "AccountBalance"])
sample_prediction = model.predict(sample_data)
print(f"Sample Prediction (Age=35, Gender=1, Balance=50000): {sample_prediction[0]}")