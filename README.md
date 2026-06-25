# 💸 Loan Approval Prediction System

An interactive machine learning-powered web application that evaluates loan applicants' creditworthiness in real-time. Built using Python, Scikit-Learn, and Streamlit.

---

## 🚀 Features & Project Overview

1. **Exploratory Data Analysis ([loan.ipynb](file:///Users/yash/Desktop/Loan_System/loan.ipynb))**: Jupyter Notebook containing missing value inspections, exploratory charts (pie charts, bar plots, boxplots), correlation heatmaps, and initial model evaluation (Logistic Regression, KNN, Naive Bayes).
2. **Automated Pipeline ([train_model.py](file:///Users/yash/Desktop/Loan_System/train_model.py))**: A script that fits missing value imputers, label encoders, one-hot encoders, scales features, applies non-linear feature engineering (squared terms), trains the final **Logistic Regression** model, and exports the full pipeline to `loan_model_pipeline.joblib`.
3. **Interactive Web App ([app.py](file:///Users/yash/Desktop/Loan_System/app.py))**: A premium Streamlit dashboard with a two-column input layout for real-time predictions. The app outputs loan approved/denied status cards with matching confidence probability metrics.

---

## 📊 Model Evaluation Results

The final **Logistic Regression** classifier is trained on [loanapp.csv](file:///Users/yash/Desktop/Loan_System/loanapp.csv) (80/20 train-test split) and achieves the following metrics on test data:

*   **Accuracy**: `87.50%`
*   **Precision**: `79.03%`
*   **Recall**: `80.33%`
*   **F1 Score**: `79.67%`

---

## 📁 Repository Structure

*   `loan.ipynb` - Jupyter Notebook for research and exploration.
*   `loanapp.csv` - Raw dataset containing 1,000 applicant loan records.
*   `train_model.py` - Pipeline training and serialization script.
*   `app.py` -  Application dashboard code.
*   `loan_model_pipeline.joblib` - Serialized model pipeline dictionary (imputers, encoders, scaler, classifier, and column names metadata).

---

## 🛠️ Getting Started

### 1. Prerequisites
Ensure you have Python installed, then install the required dependencies:
```bash
pip install streamlit joblib scikit-learn pandas numpy
```

### 2. Train the Model and Save the Pipeline
Run the training script to fit all preprocessors, train the Logistic Regression model, and generate `loan_model_pipeline.joblib`:
```bash
python3 train_model.py
```

### 3. Launch the Web Application
Start the Streamlit server locally:
```bash
streamlit run app.py --server.address 127.0.0.1 --server.headless true
```
Open your browser and navigate to the address printed in the terminal (usually `http://127.0.0.1:8501`).
