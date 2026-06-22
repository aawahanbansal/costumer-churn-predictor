import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

st.title("Airline Passenger Satisfaction Predictor")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
# Remove unnecessary columns
    df = df.drop(columns=['id', 'satisfaction'], errors='ignore')
# Remove unnamed index column if present
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Apply saved encoders
    for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
        df[col] = encoders[col].transform(df[col])

    feature_cols = [
        'Gender', 'Customer Type', 'Age',
        'Type of Travel', 'Class', 'Flight Distance',
        'Inflight wifi service',
        'Departure/Arrival time convenient',
        'Ease of Online booking',
        'Gate location',
        'Food and drink',
        'Online boarding',
        'Seat comfort',
        'Inflight entertainment',
        'On-board service',
        'Leg room service',
        'Baggage handling',
        'Checkin service',
        'Inflight service',
        'Cleanliness',
        'Departure Delay in Minutes',
        'Arrival Delay in Minutes'
    ]
    
    for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
        st.write(df[col].unique())
    
    X = df[feature_cols]
    df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(
    df['Arrival Delay in Minutes'].median())

    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    df["Prediction"] = predictions
    df["Prediction"] = df["Prediction"].map({
        0: "Neutral/Dissatisfied",
        1: "Satisfied"
    })

    df["Satisfaction Probability"] = probabilities

    st.success("Prediction Complete!")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Results",
        csv,
        "predictions.csv",
        "text/csv"
    )
