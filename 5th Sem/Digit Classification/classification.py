# MNIST Binary Classification (0 vs 1) using Neural Network + Confusion Matrix + Camera Input

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import cv2

# 1. Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Filter only digits 0 and 1
train_filter = np.where((y_train == 0) | (y_train == 1))
test_filter = np.where((y_test == 0) | (y_test == 1))

x_train, y_train = x_train[train_filter], y_train[train_filter]
x_test, y_test = x_test[test_filter], y_test[test_filter]

# 3. Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# 4. Build binary classifier model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')   
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 5. Train model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# 6. Evaluate and predict
y_pred = model.predict(x_test)
y_pred_classes = (y_pred > 0.5).astype("int32").flatten()

acc = accuracy_score(y_test, y_pred_classes)
print(f"\nTest Accuracy: {acc * 100:.2f}%")

# 7. Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix (0 vs 1)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# 8. Misclassified images
misclassified_idx = np.where(y_pred_classes != y_test)[0]
print(f"\nNumber of misclassified images: {len(misclassified_idx)}")

plt.figure(figsize=(8, 8))
for i, idx in enumerate(misclassified_idx[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"True: {y_test[idx]}, Pred: {y_pred_classes[idx]}")
    plt.axis('off')
plt.suptitle("Misclassified Digits (0 vs 1)")
plt.show()

model.save("mnist_binary_0_1_classifier.h5")
