import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

data = pd.read_csv(r"C:\Users\hpo\Downloads\Iris (1).csv")

print(data.head())

X = data.drop("Species", axis=1)
y = data["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred,
                      labels=["Iris-setosa",
                              "Iris-versicolor",
                              "Iris-virginica"])
print("Confusion Matrix:\n", cm)

TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

print("TP:", TP)
print("FP:", FP)
print("TN:", TN)
print("FN:", FN)

accuracy = (TP + TN) / (TP + TN + FP + FN)
print("Accuracy:", accuracy)

error_rate = (FP + FN) / (TP + TN + FP + FN)
print("Error Rate:", error_rate)

precision = TP / (TP + FP)
print("Precision:", precision)

recall = TP / (TP + FN)
print("Recall:", recall)
