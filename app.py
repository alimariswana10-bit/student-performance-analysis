import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Application title and description
st.title("Student Data Analysis")
st.write("Explore student study hours, attendance, and marks.")

# Load CSV dataset
df = pd.read_csv("sample_students.csv")

# Display dataset
st.subheader("Student Dataset")

display_columns = st.multiselect(
    "Select columns to display",
    df.columns.tolist(),
    default=df.columns.tolist()
)

st.dataframe(df[display_columns], use_container_width=True)

# Summary statistics
st.subheader("Summary Statistics")
st.write(df.describe())

# Bar chart
st.subheader("Student Marks Chart")

fig, ax = plt.subplots()
ax.bar(df["Student"], df["Marks"])
ax.set_xlabel("Student")
ax.set_ylabel("Marks")
ax.set_title("Student Performance")

st.pyplot(fig)
plt.close(fig)

# Interactive filter
st.subheader("Filter Students by Marks")

minimum = st.slider(
    "Select minimum marks",
    min_value=0,
    max_value=100,
    value=50
)

filtered_df = df[df["Marks"] >= minimum]

st.write("Number of students:", len(filtered_df))
st.dataframe(filtered_df, use_container_width=True)


# Machine Learning Prediction
st.subheader("Predict Student Marks")

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=90.0
)

if st.button("Predict Marks"):
    from sklearn.linear_model import LinearRegression

    model_df = pd.read_csv("cleaned_students.csv")
    X = model_df[["Study_Hours", "Attendance"]]
    y = model_df["Marks"]

    model = LinearRegression()
    model.fit(X, y)

    input_data = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance]
    })

    prediction = model.predict(input_data)[0]
    prediction = max(0, min(100, prediction))

    st.success(f"Predicted Marks: {prediction:.2f}")
    st.caption(
        "This is an educational estimate based on a very small sample dataset."
    )
