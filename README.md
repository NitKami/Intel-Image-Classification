# 🏞️ Intel Image Classification Project

A Deep Learning project using Convolutional Neural Networks (CNN) to classify natural scenes into six categories:

🏢 Buildings | 🌳 Forest | 🧊 Glacier | ⛰️ Mountain | 🌊 Sea | 🛣️ Street

---

## 📚 About the Dataset

The dataset contains natural scene images divided into six classes:

- 📁 **Buildings**
- 📁 **Forest**
- 📁 **Glacier**
- 📁 **Mountain**
- 📁 **Sea**
- 📁 **Street**

📌 **Dataset Source:**  
Download it from Kaggle:  
👉 [Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

**Note:** Dataset is not uploaded to GitHub due to size limitations.

---

## 🛠 Project Workflow

1. **Data Preprocessing**  
   - Images rescaled (`1./255`) for faster convergence.
   - Train-test split handled with `ImageDataGenerator`.

2. **Model Building**  
   - CNN architecture with 3 convolutional blocks.
   - ReLU activation, MaxPooling layers.
   - Dense layers with Dropout for regularization.

3. **Model Training**  
   - Trained for 15 epochs using `Adam` optimizer.
   - Loss function: `Categorical Crossentropy`.

4. **Model Saving**  
   - Model saved as `intel_image_classification_model.h5` for future use.

5. **Image Prediction**  
   - `classify_image.py` script provided to predict any custom image!

---

## 🚀 How to Run the Project Locally

1. Clone the repository
bash
git clone https://github.com/YourUsername/Intel-Image-Classification.git
cd Intel-Image-Classification
2. Install required packages
bash
Copy
Edit
pip install -r requirements.txt
3. To Train the Model (Optional)
bash
Copy
Edit
python train_model.py
(Skip if you want to use the already trained .h5 model.)

4. To Predict an Image
bash
Copy
Edit
python classify_image.py
Enter the image path when prompted!

🎯 Project Structure
bash
Copy
Edit
Intel-Image-Classification/
├── train_model.py                  # Model training script
├── classify_image.py                # Image prediction script
├── intel_image_classification_model.h5  # Trained CNN model
├── requirements.txt                 # Required libraries
└── README.md                        # Project overview
📦 Tech Stack Used
Python 3.10+

TensorFlow / Keras

NumPy

Pillow

✨ Future Enhancements
✅ Implement Transfer Learning with MobileNetV2

✅ Deploy model as a web app (Streamlit / Flask)

✅ Hyperparameter tuning for improved accuracy

✅ Model Explainability (GradCAM visualization)

💡 Key Learnings
How to build a CNN from scratch.

How to handle image data in deep learning.

How to save and reuse trained models for predictions.

👨‍💻 Author
Made with ❤️ by Nishnat