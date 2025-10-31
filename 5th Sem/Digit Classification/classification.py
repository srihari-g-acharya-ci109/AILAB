# MNIST Digit Classification using Neural Network + Confusion Matrix + Camera Input

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

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

model = Sequential([
    Flatten(input_shape=(28, 28)),  # Converts 2D images to 1D
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 10 output classes (digits 0–9)
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train_cat, epochs=5, batch_size=32, validation_split=0.1)

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
acc = accuracy_score(y_test, y_pred_classes)
print(f"\nTest Accuracy: {acc * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

misclassified_idx = np.where(y_pred_classes != y_test)[0]
print(f"\nNumber of misclassified images: {len(misclassified_idx)}")

plt.figure(figsize=(10, 10))
for i, idx in enumerate(misclassified_idx[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"True: {y_test[idx]}, Pred: {y_pred_classes[idx]}")
    plt.axis('off')
plt.suptitle("Misclassified Digits")
plt.show()

model.save("mnist_digit_classifier.h5")




# 11. Optional: Capture from camera and predict
choice = input("\nPress 1 to open camera and predict a digit (or any other key to exit): ")

if choice == '1':
    cap = cv2.VideoCapture(0)
    print("Press 'c' to capture an image, or 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Camera - Show a Digit", frame)

        key = cv2.waitKey(1)
        if key == ord('c'):  
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (28, 28))
            gray = cv2.bitwise_not(gray)  
            gray = gray / 255.0
            pred = model.predict(gray.reshape(1, 28, 28))
            digit = np.argmax(pred)
            print(f"Predicted Digit: {digit}")
            cv2.imshow("Processed Image", gray)
            cv2.waitKey(0)
        elif key == ord('q'):  
            break

    cap.release()
    cv2.destroyAllWindows()
