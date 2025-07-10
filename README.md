# 🧠 Neural Networks with TensorFlow/Keras  


---

## 🎯 Objective  
To explore the implementation and training of neural networks using TensorFlow and Keras. This lab includes:
- Building a simple digit recognition model using the MNIST dataset.
- Constructing a convolutional neural network (CNN) for custom face recognition.

---

## 📁 Code Files

### 🔹 `mnist_neural_net.py`
This script implements a basic **feedforward neural network** trained on the MNIST digit dataset.

#### 📌 Key Components:
- **Layers:** Flatten, Dense (ReLU), Dropout, Dense (Softmax)
- **Optimizer:** Adam  
- **Loss Function:** Sparse Categorical Crossentropy  
- **Epochs:** 5  
- **Validation:** Uses test set

#### 📈 Output:
- Final test accuracy printed
- Prediction vs actual value for first test sample
- Accuracy and loss plots saved to `outputs/`

---

### 🔹 `face_recognition_cnn.py`
This script builds a **Convolutional Neural Network** (CNN) to distinguish the user's face from others.

#### 📌 Key Components:
- **Data:** Uses images extracted from my own pictures from folder`Nofil.zip` and from other pictures that is in the folder `NotNofil.zip`
- **Layers:** Conv2D, MaxPooling2D, Flatten, Dense, Dropout
- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Epochs:** 20  
- **Regularization:** Dropout

#### 📈 Output:
- Trained model saved as `nofil_face_recognizer.keras`
- Accuracy curve saved to `outputs/`

---





## 🧰 Requirements

Install dependencies using:
```bash
pip install tensorflow matplotlib
```

---

## 🔁 How to Run

```bash
python mnist_neural_net.py
python face_recognition_cnn.py
```

Ensure zipped face datasets (`Nofil.zip`, `NotNofil.zip`) are placed at the correct paths before running the face recognition script.

---

## 📌 Notes
- The MNIST model achieves ~98% accuracy.
- Face model reaches 100% accuracy early due to simple and balanced dataset
