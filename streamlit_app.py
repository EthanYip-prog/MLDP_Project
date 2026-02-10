import joblib
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Airline Satisfaction Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

## Load trained model (tuned Gradient Boosting)
model = joblib.load("flight_satisfaction_best_rs_gb_model.pkl")

st.title("Airline Passenger Satisfaction")
st.caption("Adjust inputs below — results update instantly as you change values.")

# Subtle styling for the prediction panel only
st.markdown(
    """
<style>
.prediction-card {
  padding: 16px 18px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.04);
  box-shadow: 0 8px 22px rgba(0, 0, 0, 0.18);
}
.prediction-label {
  font-size: 0.9rem;
  opacity: 0.8;
  margin-bottom: 6px;
}
.prediction-value {
  font-size: 1.4rem;
  font-weight: 700;
  margin-bottom: 8px;
}
.prediction-prob {
  font-size: 0.95rem;
  opacity: 0.85;
}
</style>
""",
    unsafe_allow_html=True,
)

## Categorical input options
genders = ["Female", "Male"]
customer_types = ["Loyal Customer", "disloyal Customer"]
travel_types = ["Business travel", "Personal Travel"]
classes = ["Business", "Eco", "Eco Plus"]

left, right = st.columns([1.25, 1])

with left:
    st.subheader("Passenger Details")
    gender_selected = st.selectbox("Gender", genders)
    customer_type_selected = st.selectbox("Customer Type", customer_types)
    travel_type_selected = st.selectbox("Type of Travel", travel_types)
    class_selected = st.selectbox("Class", classes)

    st.subheader("Trip Context")
    age = st.slider("Age", min_value=7, max_value=85, value=30)
    flight_distance = st.number_input("Flight Distance", min_value=0, max_value=5000, value=1000)
    dep_delay = st.number_input("Departure Delay in Minutes", min_value=0, max_value=1000, value=0)
    arr_delay = st.number_input("Arrival Delay in Minutes", min_value=0, max_value=1000, value=0)

    st.subheader("Service Ratings")
    wifi = st.slider("Inflight wifi service", 1, 5, 3)
    dep_arr_time = st.slider("Departure/Arrival time convenient", 1, 5, 3)
    online_booking = st.slider("Ease of Online booking", 1, 5, 3)
    gate_location = st.slider("Gate location", 1, 5, 3)
    food_drink = st.slider("Food and drink", 1, 5, 3)
    online_boarding = st.slider("Online boarding", 1, 5, 3)
    seat_comfort = st.slider("Seat comfort", 1, 5, 3)
    inflight_ent = st.slider("Inflight entertainment", 1, 5, 3)
    onboard_service = st.slider("On-board service", 1, 5, 3)
    leg_room = st.slider("Leg room service", 1, 5, 3)
    baggage_handling = st.slider("Baggage handling", 1, 5, 3)
    checkin_service = st.slider("Checkin service", 1, 5, 3)
    inflight_service = st.slider("Inflight service", 1, 5, 3)
    cleanliness = st.slider("Cleanliness", 1, 5, 3)

with right:
    st.subheader("Prediction Result")
    input_data = {
        "Gender": gender_selected,
        "Customer Type": customer_type_selected,
        "Type of Travel": travel_type_selected,
        "Class": class_selected,
        "Age": age,
        "Flight Distance": flight_distance,
        "Inflight wifi service": wifi,
        "Departure/Arrival time convenient": dep_arr_time,
        "Ease of Online booking": online_booking,
        "Gate location": gate_location,
        "Food and drink": food_drink,
        "Online boarding": online_boarding,
        "Seat comfort": seat_comfort,
        "Inflight entertainment": inflight_ent,
        "On-board service": onboard_service,
        "Leg room service": leg_room,
        "Baggage handling": baggage_handling,
        "Checkin service": checkin_service,
        "Inflight service": inflight_service,
        "Cleanliness": cleanliness,
        "Departure Delay in Minutes": dep_delay,
        "Arrival Delay in Minutes": arr_delay,
    }

    df_input = pd.DataFrame([input_data])

    # One-hot encode categoricals (same as training: drop_first=True)
    df_input = pd.get_dummies(
        df_input,
        columns=["Gender", "Customer Type", "Type of Travel", "Class"],
        drop_first=True,
    )

    # Align to training feature columns
    df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predict
    pred = model.predict(df_input)[0]
    label = "Satisfied" if pred == 1 else "Neutral or Dissatisfied"

    prob = None
    if hasattr(model, "predict_proba"):
        class_index = list(model.classes_).index(1)
        prob = model.predict_proba(df_input)[0][class_index]

    prob_text = f"{prob:.2f}" if prob is not None else "N/A"
    st.markdown(
        f"""
<div class="prediction-card">
  <div class="prediction-label">Predicted Satisfaction</div>
  <div class="prediction-value">{label}</div>
  <div class="prediction-prob">Probability of Satisfied: {prob_text}</div>
</div>
""",
        unsafe_allow_html=True,
    )
