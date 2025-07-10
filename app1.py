import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit App Title
st.title("🏡 California Housing Price Prediction")
st.markdown("### Predicting Median House Value using Median Income")
st.write("This app uses Linear Regression to predict median house values in California based on district-level features.")

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Data Preview
    st.subheader("🔍 Raw Data Preview")
    st.dataframe(df.head())

    # Preprocessing
    df = df[['median_income', 'median_house_value']]
    df.dropna(inplace=True)

    st.subheader("📊 Data Summary")
    st.write(df.describe())

    # Correlation
    st.subheader("📈 Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)

    # Scatter Plot
    st.subheader("🔍 Income vs House Value (Before Outlier Removal)")
    fig_scatter1, ax_scatter1 = plt.subplots()
    ax_scatter1.scatter(df['median_income'], df['median_house_value'])
    ax_scatter1.set_xlabel("Median Income")
    ax_scatter1.set_ylabel("Median House Value")
    st.pyplot(fig_scatter1)

    # Box Plot
    st.subheader("📦 House Value Distribution")
    fig_box, ax_box = plt.subplots()
    sns.boxplot(y='median_house_value', data=df, ax=ax_box)
    st.pyplot(fig_box)

    # Jointplot (Before Outlier Removal)
    st.subheader("📉 Regression Plot (Before Outliers Removed)")
    fig_joint_before = sns.jointplot(x='median_income', y='median_house_value', data=df, kind='reg')
    st.pyplot(fig_joint_before.fig)

    # Remove Outliers
    df = df[df['median_house_value'] <= 500000]

    # Jointplot (After Outlier Removal)
    st.subheader("✅ Regression Plot (After Outliers Removed)")
    fig_joint_after = sns.jointplot(x='median_income', y='median_house_value', data=df, kind='reg')
    st.pyplot(fig_joint_after.fig)

    # Prepare Data
    X = df[['median_income']]
    y = df['median_house_value']

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Plot Regression Line
    st.subheader("📈 Regression Line Fit")
    line = model.coef_[0] * X + model.intercept_
    fig_line, ax_line = plt.subplots()
    ax_line.scatter(X, y)
    ax_line.plot(X, line, color='red')
    ax_line.set_xlabel('Median Income')
    ax_line.set_ylabel('Median House Value')
    ax_line.set_title('Regression Line')
    st.pyplot(fig_line)

    # Predictions
    y_pred = model.predict(X_test)
    result_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
    st.subheader("📄 Actual vs Predicted Values")
    st.dataframe(result_df.head(10))

    # Evaluation Metrics
    st.subheader("📊 Model Evaluation")
    st.write(f"**Coefficient:** {model.coef_[0]:.2f}")
    st.write(f"**Mean Squared Error:** {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

    # Custom Prediction
    st.subheader("🧮 Predict House Value for Custom Income")
    user_income = st.number_input("Enter Median Income", min_value=0.0, value=4.0, step=0.1)
    prediction = model.predict([[user_income]])
    st.success(f"💰 Predicted Median House Value: ${prediction[0]:,.2f}")
else:
    st.info("👆 Please upload a CSV file to start.")