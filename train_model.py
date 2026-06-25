import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

def main():
    print("Loading dataset...")
    df = pd.read_csv('loanapp.csv')

    # 1. Identify columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Applicant_ID is just an identifier (not useful for prediction)
    if 'Applicant_ID' in num_cols:
        num_cols.remove('Applicant_ID')
    if 'Loan_Approved' in cat_cols:
        cat_cols.remove('Loan_Approved')
        df['Loan_Approved'] = df['Loan_Approved'].fillna(df['Loan_Approved'].mode()[0])

    print("Handling missing values...")
    # Numerical Imputer
    imputer_num = SimpleImputer(strategy='mean')
    df[num_cols] = imputer_num.fit_transform(df[num_cols])

    # Categorical Imputer
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

    # 2. Encode binary categorical features
    print("Encoding target and Education_Level...")
    le_edu = LabelEncoder()
    df["Education_Level"] = le_edu.fit_transform(df["Education_Level"])
    
    le_target = LabelEncoder()
    df["Loan_Approved"] = le_target.fit_transform(df["Loan_Approved"])

    # Drop Applicant_ID
    df = df.drop("Applicant_ID", axis=1)

    # 3. One-Hot Encoding for remaining categorical columns
    ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
    print(f"One-hot encoding columns: {ohe_cols}")
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(df[ohe_cols])
    
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=df.index)
    df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

    # 4. Feature Engineering (Squaring key features)
    print("Performing feature engineering...")
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2
    df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2

    # Drop target and original engineered variables from the feature set
    X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"])
    y = df["Loan_Approved"]
    
    feature_names = X.columns.tolist()

    # 5. Train-Test Split
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 7. Model Training
    print("Training Logistic Regression model...")
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_scaled, y_train)

    # 8. Model Evaluation
    y_pred = log_model.predict(X_test_scaled)
    print("\nLOGISTIC REGRESSION EVALUATION")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 9. Save all artifacts in a single pipeline file
    pipeline = {
        'imputer_num': imputer_num,
        'imputer_cat': imputer_cat,
        'le_edu': le_edu,
        'ohe': ohe,
        'scaler': scaler,
        'model': log_model,
        'feature_names': feature_names,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'ohe_cols': ohe_cols
    }
    
    pipeline_filename = 'loan_model_pipeline.joblib'
    joblib.dump(pipeline, pipeline_filename)
    print(f"\nSaved pipeline to '{pipeline_filename}' successfully!")

if __name__ == "__main__":
    main()
