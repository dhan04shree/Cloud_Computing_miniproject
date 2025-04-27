import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Student Depression Insights", layout="wide")

# ---------- Load and Clean Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("student_depression_dataset.csv")
    # Basic cleaning
    df.columns = df.columns.str.strip()
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)  # Or use df.fillna(method='ffill') for forward fill
    df = clean_data(df)  # Clean the data
    return df

# ---------- Clean Data Function ----------
def convert_to_numeric(value):
    if isinstance(value, str):
        # Handle cases like '5-6 hours'
        if '-' in value:
            parts = value.split('-')
            try:
                return (float(parts[0]) + float(parts[1].split()[0])) / 2
            except ValueError:
                return np.nan
        # For other cases, try to extract the numeric value
        try:
            return float(value.split()[0])  # Assuming it's in the format like '5 hours'
        except ValueError:
            return np.nan  # If it cannot be converted, return NaN
    return value  # If it's already a number, return it as is

def clean_data(df):
    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Apply the conversion to the relevant columns
    df['Sleep Duration'] = df['Sleep Duration'].apply(convert_to_numeric)
    df['Work/Study Hours'] = df['Work/Study Hours'].apply(convert_to_numeric)

    # Ensure that the columns being used for correlation contain numeric values
    mental_cols = ['Academic Pressure', 'Work Pressure', 'Sleep Duration', 'Work/Study Hours', 'Financial Stress']
    for col in mental_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Force non-numeric to NaN

    return df

# ---------- EDA Summary ----------
def show_eda(df):
    st.subheader("🔍 Exploratory Data Analysis")
    st.write("**Dataset Preview**")
    st.dataframe(df.head())

    st.write("**Basic Statistics**")
    st.dataframe(df.describe())

    st.write("**Missing Values**")
    st.dataframe(df.isnull().sum())

# ---------- Insight 1: Depression by Department and Gender ----------
def insight_1(df):
    st.subheader("📊 Depression Distribution by Gender and Profession")
    
    # Check the column names
    st.write("Columns in dataset:", df.columns)

    # Ensure columns are cleaned
    df.columns = df.columns.str.strip()

    # Group by Gender and Profession
    grouped = df.groupby(["Gender", "Profession"])["Depression"].value_counts(normalize=True).unstack().fillna(0)
    
    # Reset the index to flatten the MultiIndex
    grouped_reset = grouped.reset_index()

    # Plot the data
    fig = px.bar(grouped_reset, x="Gender", y=grouped_reset.columns[2:], color="Profession", barmode='group', title="Depression Rate by Gender and Profession")
    st.plotly_chart(fig, use_container_width=True)


# ---------- Insight 2: Correlation Heatmap ----------
def insight_2(df):
    st.subheader("📊 Correlation between Mental Health Factors")
    
    # Check the column names in the dataset
    st.write("Columns in dataset:", df.columns)

    # List of columns you want to check correlation for (adjust based on your dataset)
    mental_cols = ['Academic Pressure', 'Work Pressure', 'Sleep Duration', 'Work/Study Hours', 'Financial Stress']

    # Ensure these columns exist in the dataset
    missing_cols = [col for col in mental_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        return

    # Calculate the correlation matrix
    corr = df[mental_cols].corr()

    # Display the correlation matrix as a heatmap
    fig = px.imshow(corr, title="Correlation Heatmap of Mental Health Factors", color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)


# ---------- Insight 3: Sleep vs Depression ----------
# ---------- Insight 3: Sleep vs Depression ----------
def insight_3(df):
    st.subheader("🛌 Sleep Duration vs Depression")
    
    # Convert Depression to numeric if needed
    if df['Depression'].dtype == 'object':
        df['Depression Numeric'] = df['Depression'].map({'Yes': 1, 'No': 0})
    else:
        df['Depression Numeric'] = df['Depression']

    fig = px.scatter(
        df,
        x="Sleep Duration",
        y="Depression Numeric",
        trendline="ols",
        color="Gender",
        labels={
            "Sleep Duration": "Sleep Duration (hours)",
            "Depression Numeric": "Depression (1 = Yes, 0 = No)"
        },
        title="Sleep Duration vs Depression"
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------- Insight 4: CGPA vs Depression ----------
def insight_4(df):
    st.subheader("🎓 Academic Performance vs Depression")

    fig = px.box(
        df,
        x="Depression",
        y="CGPA",
        color="Depression",
        points="all",
        title="CGPA Distribution by Depression Status"
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------- Insight 5: Study Hours vs Depression ----------
def insight_5(df):
    st.subheader("📚 Study Hours and Their Impact on Depression")

    fig = px.box(
        df,
        x="Depression",
        y="Work/Study Hours",
        color="Depression",
        points="all",
        title="Study Hours by Depression Status"
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------- Main App ----------
def main():
    st.title("🧠 Student Depression Insights Dashboard")
    st.markdown("""
        This interactive dashboard analyzes patterns in student mental health using a dataset on depression, 
        academic performance, and lifestyle factors. Explore key insights to support data-driven well-being strategies.
    """)

    df = load_data()

    # Sidebar navigation
    menu = st.sidebar.radio("Navigate", ["Overview", "EDA", "Insights"])

    if menu == "Overview":
        st.header("📁 Dataset Overview")
        st.write("Upload your own dataset below or use the default one.")
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success("Uploaded new dataset!")

        st.dataframe(df.head())

    elif menu == "EDA":
        show_eda(df)

    elif menu == "Insights":
        insight_1(df)
        st.divider()
        insight_2(df)
        st.divider()
        insight_3(df)
        st.divider()
        insight_4(df)
        st.divider()
        insight_5(df)

if __name__ == "__main__":
    main()
