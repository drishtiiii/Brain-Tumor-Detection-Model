# 🧠 Brain Tumor Detection Using Convolutional Neural Networks (CNN)

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-CNN-red?style=for-the-badge&logo=keras)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

</p>

---

# 📌 Overview

Brain tumors are among the most critical neurological disorders, where early diagnosis significantly improves treatment planning and patient outcomes. Manual interpretation of MRI scans is both time-consuming and dependent on expert radiologists. This project explores the application of **Deep Learning** using **Convolutional Neural Networks (CNNs)** to automate the detection of brain tumors from MRI images.

The model learns spatial features directly from MRI scans through multiple convolutional layers and performs binary classification to determine whether a tumor is present.

The project demonstrates the complete deep learning workflow including data preprocessing, augmentation, CNN model development, training, evaluation, prediction, and visualization using **TensorFlow/Keras**.

---

# ✨ Features

✔ Binary Brain Tumor Classification (Tumor / No Tumor)

✔ MRI Image Processing

✔ Image Data Augmentation

✔ Custom CNN Architecture

✔ Batch Normalization

✔ Max Pooling

✔ Adam Optimizer

✔ Binary Crossentropy Loss

✔ Training & Validation Accuracy Visualization

✔ Training & Validation Loss Visualization

✔ Classification Report

✔ Confusion Matrix

✔ Model Prediction on MRI Images

✔ Trained Model Saving (.h5)

---

# 🎯 Objectives

The primary objectives of this project are:

- Develop a Convolutional Neural Network capable of classifying MRI images into tumor and non-tumor categories.
- Reduce manual intervention during the preliminary screening process.
- Explore the effectiveness of deep learning techniques for medical image classification.
- Understand the complete workflow involved in developing an image classification model using TensorFlow and Keras.

---

# 🧰 Technologies Used

| Category | Technologies |
|-----------|--------------|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Computer Vision | OpenCV |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Machine Learning | Scikit-learn |
| Development Environment | Google Colaboratory / Jupyter Notebook |

---

# 📂 Dataset

The project utilizes a dataset consisting of **Brain MRI images** categorized into two classes:

- 🧠 Tumor
- 🧠 No Tumor

The images are loaded using Keras' **ImageDataGenerator**, allowing efficient preprocessing and augmentation during training.

### Data Preprocessing

The preprocessing pipeline includes:

- Image resizing to **256 × 256 pixels**
- RGB color format
- Pixel normalization using rescaling (1/255)
- Horizontal image flipping
- Vertical image flipping
- Automatic validation split
- Batch loading using ImageDataGenerator

These preprocessing techniques improve model generalization and reduce overfitting.

---

# 📁 Repository Structure

```
Brain-Tumor-Detection-Model/
│
├── README.md
├── archive_2.zip
├── brain_tumor.ipynb
└── brain_tumorDetectionUsingCNN.ipynb
```

### Repository Contents

| File | Description |
|------|-------------|
| brain_tumor.ipynb | Initial experimentation notebook containing preprocessing and exploratory work. |
| brain_tumorDetectionUsingCNN.ipynb | Complete CNN implementation including training, evaluation and prediction. |
| archive_2.zip | Dataset archive used during model development. |
| README.md | Project documentation. |

---

# 🚀 Project Workflow

```
MRI Images
      │
      ▼
Data Preprocessing
      │
      ▼
Data Augmentation
      │
      ▼
CNN Model
      │
      ▼
Model Training
      │
      ▼
Evaluation
      │
      ▼
Prediction
```

---

## 📖 Table of Contents

- Overview
- Features
- Objectives
- Technologies Used
- Dataset
- Repository Structure
- Project Workflow
- Installation
- CNN Architecture
- Model Training
- Results
- Future Improvements
- License
- Author

---

# ⚙️ Installation

Follow these steps to run the project locally.

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/drishtiiii/Brain-Tumor-Detection-Model.git
cd Brain-Tumor-Detection-Model
```

---

## 2️⃣ Create a Virtual Environment (Optional)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If you are using **Google Colab**, simply install any missing libraries using:

```python
!pip install tensorflow opencv-python
```

---

## 4️⃣ Prepare the Dataset

Download the Brain MRI dataset and place it in the appropriate directory.

The notebooks expect the dataset to be extracted before training begins.

Example folder structure:

```
Dataset/

├── yes/
│      image1.jpg
│      image2.jpg
│      ...
│
└── no/
       image1.jpg
       image2.jpg
       ...
```

---

## 5️⃣ Run the Notebook

Open either notebook using Jupyter Notebook or Google Colab.

```
brain_tumor.ipynb
```

or

```
brain_tumorDetectionUsingCNN.ipynb
```

Execute all cells sequentially.

---

# 🧠 CNN Model Architecture

This project implements a **custom Convolutional Neural Network (CNN)** designed specifically for binary MRI image classification.

The model automatically extracts spatial features from MRI scans and progressively learns complex representations of brain tissue.

## Model Pipeline

```
Input Image
      │
      ▼
Convolution Layer
      │
      ▼
Batch Normalization
      │
      ▼
Max Pooling
      │
      ▼
Convolution Layer
      │
      ▼
Batch Normalization
      │
      ▼
Max Pooling
      │
      ▼
Convolution Layer
      │
      ▼
Batch Normalization
      │
      ▼
Max Pooling
      │
      ▼
Convolution Layer
      │
      ▼
Batch Normalization
      │
      ▼
Max Pooling
      │
      ▼
Flatten
      │
      ▼
Dense (128)
      │
      ▼
Dense (128)
      │
      ▼
Dense (64)
      │
      ▼
Output Layer
```

---

## Layer Details

### Input Layer

- Image Size: **256 × 256**
- RGB Channels
- Normalized Pixel Values

---

### Convolution Layers

The CNN contains multiple convolutional blocks responsible for learning image features such as:

- Edges
- Texture
- Shapes
- Tumor boundaries
- Brain structures

Each convolution block is followed by:

- Batch Normalization
- Max Pooling

This improves convergence while reducing spatial dimensions.

---

### Batch Normalization

Batch Normalization is used after convolution layers to:

- Speed up training
- Improve stability
- Reduce internal covariate shift
- Allow higher learning rates

---

### Max Pooling

Max Pooling reduces image dimensions while preserving important features.

Benefits include:

- Reduced computation
- Lower memory usage
- Improved generalization
- Reduced overfitting

---

### Fully Connected Layers

After flattening the extracted features, multiple Dense layers perform the final classification.

Dense Layers:

- 128 Neurons
- 128 Neurons
- 64 Neurons

These layers learn high-level feature relationships before making the final prediction.

---

### Output Layer

The final layer contains:

- **1 neuron**
- **Sigmoid activation**

Output values:

- **0 → No Tumor**
- **1 → Tumor**

Since this is a binary classification problem, **Binary Crossentropy** is used as the loss function.

---

# ⚡ Model Compilation

The CNN model is compiled using:

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Evaluation Metric | Accuracy |

The Adam optimizer was selected because of its adaptive learning capabilities and efficient convergence during deep learning model training.

---

# 🎯 Training Configuration

The model is trained using TensorFlow/Keras with the following settings.

| Parameter | Value |
|-----------|-------|
| Epochs | 40 |
| Image Size | 256 × 256 |
| Color Mode | RGB |
| Validation Data | Yes |
| Shuffle | Enabled |

The model is trained using mini-batches generated through **ImageDataGenerator**, enabling efficient preprocessing and augmentation during training.

---

# 🔄 Data Augmentation

To improve model robustness and reduce overfitting, the following augmentation techniques are applied:

- Image Rescaling (1/255)
- Horizontal Flip
- Vertical Flip
- Validation Split

These transformations expose the model to varied image orientations while preserving important anatomical structures.

---

# 💾 Model Saving

After training, the CNN model is saved for future inference.

```python
model.save("Modelcnn.h5")
```

The saved model can later be loaded without retraining.

```python
from tensorflow.keras.models import load_model

model = load_model("Modelcnn.h5")
```

---

# ▶️ Usage

After training completes:

1. Load the trained model.
2. Preprocess the MRI image.
3. Pass the image through the CNN.
4. Obtain the prediction probability.
5. Convert probability into a binary prediction.

The notebook also demonstrates prediction on sample MRI images and visualizes the predicted and actual labels for qualitative evaluation.

---

# 📊 Model Training & Evaluation

The CNN model was trained using TensorFlow/Keras over **40 epochs** on preprocessed MRI images. During training, the model learned hierarchical spatial features through successive convolution and pooling operations while monitoring performance on a validation dataset.

The training process included visualization of both accuracy and loss metrics, allowing the learning behaviour of the model to be observed across epochs.

---

# 📈 Training Performance

The training history indicates continuous improvement in model performance throughout the training process.

## 📈 Training Accuracy

The training accuracy improved consistently throughout the 40 training epochs. Both the training and validation curves indicate that the CNN effectively learned meaningful features from MRI images and converged after approximately 20–25 epochs.

<p align="center">
  <img src="accuracy.png" alt="Training Accuracy" width="750">
</p>

---

## 📉 Training Loss

The loss curves demonstrate the optimization process of the network. Training loss steadily decreased over time, while validation loss showed minor fluctuations before stabilizing in the later epochs, indicating effective learning.

<p align="center">
  <img src="loss.png" alt="Training Loss" width="750">
</p>

---

## 📊 Classification Report

The trained model was evaluated using precision, recall, F1-score, and overall accuracy.

| Class | Precision | Recall | F1-Score |
|--------|----------:|-------:|---------:|
| No Tumor | 0.62 | 0.56 | 0.59 |
| Tumor | 0.50 | 0.57 | 0.53 |

**Overall Accuracy:** **56%**

> *These values are taken directly from the notebook evaluation output.*

---

## 🔲 Confusion Matrix

The confusion matrix provides a detailed view of the model's predictions by comparing actual labels with predicted labels.

```
[[5 4]
 [3 4]]
```

<p align="center">
  <img src="confusion matrix.png" alt="Confusion Matrix" width="550">
</p>

---

## 🖼️ Sample Predictions

The notebook demonstrates predictions on unseen MRI scans. Each prediction displays both the predicted label and the corresponding ground truth, allowing visual verification of the model's performance.

<p align="center">
  <img src="predictions.png" alt="Sample Predictions" width="850">
</p>

The prediction workflow consists of:

1. Loading the trained CNN model.
2. Preprocessing the MRI image.
3. Performing inference using `model.predict()`.
4. Converting prediction probabilities into binary labels.
5. Comparing predicted labels with the actual class.

---

# 💡 Key Learnings

During the development of this project, several important deep learning concepts were explored:

- Medical image preprocessing
- Image augmentation techniques
- Convolutional Neural Networks (CNN)
- Batch Normalization
- Max Pooling
- Binary Image Classification
- Adam Optimizer
- Binary Crossentropy Loss
- Model Evaluation
- Classification Metrics
- Confusion Matrix Analysis
- TensorFlow/Keras model development

---

# ⚠ Limitations

Although the project demonstrates an end-to-end deep learning workflow for MRI image classification, there are several opportunities for improvement.

Current limitations include:

- Binary classification only (Tumor / No Tumor)
- Notebook-based implementation
- No web interface for inference
- Limited evaluation metrics
- Dataset size may affect generalization
- No explainability techniques such as Grad-CAM

These limitations provide opportunities for future enhancements and experimentation.

---

# 🚀 Future Improvements

This project provides a strong foundation for brain tumor classification using Convolutional Neural Networks. Several enhancements can further improve its performance, usability, and scalability.

## Planned Improvements

- Implement Transfer Learning using pretrained architectures such as VGG16, ResNet50, EfficientNet, or DenseNet.
- Expand the dataset to improve model generalization.
- Perform multiclass classification for different tumor types.
- Integrate Grad-CAM or other explainability techniques to visualize model decisions.
- Develop a web application using Streamlit or Flask for real-time predictions.
- Deploy the trained model to cloud platforms such as Render, Hugging Face Spaces, or Streamlit Community Cloud.
- Optimize hyperparameters using automated tuning techniques.
- Improve evaluation with ROC-AUC, Precision-Recall curves, and cross-validation.

---

# 📚 References

- TensorFlow Documentation
- Keras Documentation
- OpenCV Documentation
- Scikit-learn Documentation
- Brain MRI Image Dataset (Kaggle)

---

# 🤝 Contributing

Contributions are welcome.

If you would like to improve this project:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

Suggestions, bug reports, and improvements are always appreciated.

---

# 👩‍💻 Author

**Drishti Saha**

GitHub:
https://github.com/drishtiiii

If you found this repository useful, consider giving it a ⭐.

---

# 📄 License

This project is licensed under the MIT License.

Feel free to use, modify, and distribute this project for educational and research purposes.

See the LICENSE file for more details.

---

# ⭐ Support

If you like this project,

⭐ Star the repository

🍴 Fork it

📢 Share it with others

Thank you for visiting this repository!

---

