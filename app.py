import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import io

st.set_page_config(page_title="Credit Card Fraud EDA", layout="wide")
st.title("💳 Credit Card Fraud Detection - Exploratory Data Analysis")

# Upload ZIP file
uploaded_zip = st.file_uploader("Upload your `creditcard.csv.zip` file", type=["zip"])

if uploaded_zip:
    # Extract the CSV file
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
        if not csv_files:
            st.error("No CSV found inside the ZIP file.")
            st.stop()

        csv_name = csv_files[0]

        with zip_ref.open(csv_name) as csv_file:
            df = pd.read_csv(csv_file)

    st.success("File successfully loaded!")

    st.subheader("Preview of Dataset")
    st.dataframe(df.head())

    st.subheader("Basic Information")
    st.write("**Shape:**", df.shape)
    st.write("**Missing Values:**")
    st.write(df.isnull().sum())

    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.code(info_str)

    st.write("**Statistical Summary:**")
    st.dataframe(df.describe())

    # Fraud vs Non-Fraud count
    st.subheader("Fraud vs Non-Fraud Transactions")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Class', data=df, ax=ax)
    ax.set_title("Fraud vs Non-Fraud Transactions")
    st.pyplot(fig)

    fraud_ratio = df['Class'].value_counts(normalize=True) * 100
    st.write("**Class Distribution (%):**")
    st.write(fraud_ratio)

    # Transaction Amount Distribution
    st.subheader("Transaction Amount Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['Amount'], bins=50, kde=True, ax=ax)
    ax.set_title('Transaction Amount Distribution')
    st.pyplot(fig)

    # Transaction Time Distribution
    st.subheader("Transaction Time Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['Time'], bins=50, kde=True, ax=ax)
    ax.set_title('Transaction Time Distribution')
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(14, 12))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)

    # Fraud vs Non-Fraud Amount Distribution
    fraud = df[df['Class'] == 1]
    non_fraud = df[df['Class'] == 0]

    st.subheader("Transaction Amount: Fraud vs Non-Fraud")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(non_fraud['Amount'], label='Non-Fraud', bins=50, alpha=0.6)
    sns.histplot(fraud['Amount'], label='Fraud', bins=50, alpha=0.6, color='red')
    ax.legend()
    ax.set_title("Transaction Amount: Fraud vs Non-Fraud")
    st.pyplot(fig)

    # Class imbalance pie chart
    st.subheader("Class Imbalance")
    fraud_count = df['Class'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(fraud_count, labels=['Non-Fraud', 'Fraud'], autopct='%1.2f%%')
    ax.set_title('Class Imbalance')
    st.pyplot(fig)

    # Boxplot
    st.subheader("Boxplot: Amount by Class")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='Class', y='Amount', data=df, ax=ax)
    st.pyplot(fig)

    # Outlier detection for Amount
    Q1 = df['Amount'].quantile(0.25)
    Q3 = df['Amount'].quantile(0.75)
    IQR = Q3 - Q1
    if IQR == 0:
        outliers_amount = pd.DataFrame()
    else:
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_amount = df[(df['Amount'] < lower_bound) | (df['Amount'] > upper_bound)]

    st.write("**Number of outliers in Amount:**", outliers_amount.shape[0])

    # Outlier count per feature
    st.subheader("Outlier Counts per Feature (IQR Method)")
    numeric_features = df.drop(columns=['Class']).select_dtypes(include=np.number).columns
    outlier_counts = {}

    for feature in numeric_features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            outlier_counts[feature] = 0
            continue

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        outlier_counts[feature] = len(outliers)

    outlier_df = pd.DataFrame(list(outlier_counts.items()), columns=['Feature', 'Num_Outliers'])
    st.dataframe(outlier_df.sort_values(by='Num_Outliers', ascending=False))

    # Outlier comparison between fraud & non fraud
    st.subheader("Outlier Comparison: Fraud vs Non-Fraud")
    outlier_summary = {}

    for feature in numeric_features:
        outlier_summary[feature] = {"Fraud": 0, "Non-Fraud": 0}

        # Fraud
        Q1_f, Q3_f = fraud[feature].quantile([0.25, 0.75])
        IQR_f = Q3_f - Q1_f

        # Non-Fraud
        Q1_nf, Q3_nf = non_fraud[feature].quantile([0.25, 0.75])
        IQR_nf = Q3_nf - Q1_nf

        if IQR_f != 0:
            lower_f, upper_f = Q1_f - 1.5 * IQR_f, Q3_f + 1.5 * IQR_f
            outlier_summary[feature]["Fraud"] = len(fraud[(fraud[feature] < lower_f) | (fraud[feature] > upper_f)])

        if IQR_nf != 0:
            lower_nf, upper_nf = Q1_nf - 1.5 * IQR_nf, Q3_nf + 1.5 * IQR_nf
            outlier_summary[feature]["Non-Fraud"] = len(non_fraud[(non_fraud[feature] < lower_nf) | (non_fraud[feature] > upper_nf)])

    st.dataframe(pd.DataFrame(outlier_summary).T.sort_values(by="Fraud", ascending=False))

else:
    st.info("Please upload the ZIP file containing `creditcard.csv` to begin.")
