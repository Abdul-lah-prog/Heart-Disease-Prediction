# Heart-Disease-Prediction
Project Overview:
This project focuses on developing a robust machine learning-based system to detect the presence of heart disease in patients based on clinical data. The goal was to enhance prediction accuracy and provide real-time access to the model via a deployed API using FastAPI.

**Dataset & Features:**
The dataset includes patient-level clinical information such as:

Age

Resting Blood Pressure

Cholesterol Levels

Max Heart Rate Achieved

Chest Pain Type

Fasting Blood Sugar

ST Depression

Thalassemia

And more

These features serve as inputs to the machine learning models to predict the likelihood of heart disease.

**Model Training:**
Initially, six classical machine learning algorithms were trained and evaluated:

K-Nearest Neighbors (KNN)

Naive Bayes

Decision Tree

Random Forest

Support Vector Machine (SVM)

Linear Discriminant Analysis (LDA)

Despite good performance, the models achieved a maximum accuracy of 84% on the validation data.

**Dimensionality Reduction with PCA:**
To further boost model performance, Principal Component Analysis (PCA) was applied:

PCA reduced feature dimensionality while retaining the most informative components.

This transformation helped eliminate noise and redundancy in the data.

As a result of applying PCA:

The model's accuracy improved from 84% to 88%.

Overfitting was reduced, and model generalization improved.

**Deployment with FastAPI:**
To make the model accessible for real-time prediction:

A FastAPI web server was created.

The trained model and PCA transformer were saved using joblib.

The /predict endpoint accepts patient data via POST requests and returns prediction results in JSON format.

Example Input:
json
Copy
Edit
{
  "age": 52,
  "chol": 240,
  "trestbps": 130,
  "thalach": 172,
  "cp": 1,
  "fbs": 0,
  "oldpeak": 1.4,
  "thal": 2
}
Output:
json
Copy
Edit
{
  "prediction": 1,
  "label": "Heart Disease Detected"
}
The FastAPI application is interactive through Swagger UI (http://127.0.0.1:8000/docs), allowing doctors or users to easily input test data and get immediate results.

**Conclusion:**
This project demonstrates how combining multiple machine learning techniques with PCA and deploying the best-performing model via FastAPI can result in a practical, accurate, and accessible solution for heart disease prediction. It can assist healthcare providers in early diagnosis and potentially save lives through timely intervention.
