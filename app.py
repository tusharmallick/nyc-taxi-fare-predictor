"""
NYC Taxi Fare Prediction — Streamlit app
Lab Assignment 02 | Deep Feedforward Neural Network

Run locally:        streamlit run app.py
Streamlit Cloud:    this file is the entrypoint (see README for deploy steps)
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from tensorflow import keras

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="NYC Taxi Fare Predictor", page_icon="🚕", layout="centered")

# ---------------------------------------------------------------------------
# Load artifacts (same folder as this file) — cached so they load only once
# ---------------------------------------------------------------------------
MODEL_PATH = "final_taxi_fare_model.keras"
SCALER_PATH = "scaler.pkl"


@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


model, scaler = load_artifacts()

# MUST match the FEATURES list used during training, in the same order.
FEATURES = [
    "trip_distance_km", "euclidean_dist", "lat_diff", "lon_diff",
    "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
    "passenger_count", "hour", "dayofweek", "month", "year",
    "is_weekend", "is_rush_hour", "distance_per_passenger",
    "pickup_dist_jfk", "dropoff_dist_jfk", "pickup_dist_lga", "dropoff_dist_lga",
]


# ---------------------------------------------------------------------------
# Feature engineering — copied verbatim from the training notebook
# ---------------------------------------------------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometers between two (lat, lon) points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def engineer_features(data):
    data = data.copy()
    dt = data.pickup_datetime

    data["hour"] = dt.dt.hour
    data["day"] = dt.dt.day
    data["dayofweek"] = dt.dt.dayofweek
    data["month"] = dt.dt.month
    data["year"] = dt.dt.year
    data["is_weekend"] = data.dayofweek.isin([5, 6]).astype(int)
    data["is_rush_hour"] = data.hour.isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    data["trip_distance_km"] = haversine_distance(
        data.pickup_latitude, data.pickup_longitude,
        data.dropoff_latitude, data.dropoff_longitude,
    )
    data["lat_diff"] = data.dropoff_latitude - data.pickup_latitude
    data["lon_diff"] = data.dropoff_longitude - data.pickup_longitude
    data["euclidean_dist"] = np.sqrt(data.lat_diff ** 2 + data.lon_diff ** 2)
    data["distance_per_passenger"] = data.trip_distance_km / data.passenger_count.clip(lower=1)

    JFK = (-73.7781, 40.6413)
    LGA = (-73.8740, 40.7769)
    for name, (lon, lat) in [("jfk", JFK), ("lga", LGA)]:
        data[f"pickup_dist_{name}"] = haversine_distance(
            data.pickup_latitude, data.pickup_longitude, lat, lon)
        data[f"dropoff_dist_{name}"] = haversine_distance(
            data.dropoff_latitude, data.dropoff_longitude, lat, lon)

    return data


# ---------------------------------------------------------------------------
# Preset landmarks so users don't have to type coordinates
# ---------------------------------------------------------------------------
LANDMARKS = {
    "Custom (enter coordinates below)": None,
    "Times Square": (-73.9855, 40.7580),
    "Empire State Building": (-73.9857, 40.7484),
    "JFK Airport": (-73.7781, 40.6413),
    "LaGuardia Airport": (-73.8740, 40.7769),
    "Newark Airport (EWR)": (-74.1745, 40.6895),
    "Central Park": (-73.9654, 40.7829),
    "Wall Street": (-74.0089, 40.7061),
    "Brooklyn Bridge": (-73.9969, 40.7061),
    "Grand Central Terminal": (-73.9772, 40.7527),
    "Yankee Stadium": (-73.9262, 40.8296),
}

EXAMPLES = {
    "Empire State Building -> JFK Airport": (
        "Empire State Building", "JFK Airport", "2015-06-15", "18:30", 1,
    ),
    "Times Square -> Central Park": (
        "Times Square", "Central Park", "2015-03-10", "09:00", 2,
    ),
    "Wall Street -> LaGuardia Airport": (
        "Wall Street", "LaGuardia Airport", "2015-11-21", "14:15", 3,
    ),
}


def predict_fare(pickup_place, dropoff_place, pu_lon, pu_lat, do_lon, do_lat,
                  date_val, time_val, passenger_count):
    # Resolve landmark presets, falling back to the manual coordinate boxes
    if LANDMARKS.get(pickup_place):
        pu_lon, pu_lat = LANDMARKS[pickup_place]
    if LANDMARKS.get(dropoff_place):
        do_lon, do_lat = LANDMARKS[dropoff_place]

    timestamp = pd.Timestamp.combine(date_val, time_val)

    row = pd.DataFrame([{
        "pickup_longitude": float(pu_lon),
        "pickup_latitude": float(pu_lat),
        "dropoff_longitude": float(do_lon),
        "dropoff_latitude": float(do_lat),
        "pickup_datetime": timestamp,
        "passenger_count": int(passenger_count),
    }])

    row = engineer_features(row)
    X = scaler.transform(row[FEATURES].values.astype("float32"))
    fare = float(model.predict(X, verbose=0).flatten()[0])
    distance = float(row["trip_distance_km"].iloc[0])
    rush = bool(row["is_rush_hour"].iloc[0])
    weekend = bool(row["is_weekend"].iloc[0])

    return fare, distance, timestamp, rush, weekend


# ---------------------------------------------------------------------------
# Session state defaults (so the "load example" buttons can update the widgets)
# ---------------------------------------------------------------------------
defaults = {
    "pickup_place": "Empire State Building",
    "dropoff_place": "JFK Airport",
    "pu_lon": -73.9857, "pu_lat": 40.7484,
    "do_lon": -73.7781, "do_lat": 40.6413,
    "date_val": pd.Timestamp("2015-06-15").date(),
    "time_val": pd.Timestamp("2015-06-15 18:30").time(),
    "passenger_count": 1,
}
for key, val in defaults.items():
    st.session_state.setdefault(key, val)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🚕 NYC Taxi Fare Predictor")
st.caption("Deep Feedforward Neural Network | Lab Assignment 02")
st.write("Pick a landmark or enter coordinates manually, set the trip time, and get an estimated fare.")

with st.expander("Try an example"):
    cols = st.columns(len(EXAMPLES))
    for col, (label, vals) in zip(cols, EXAMPLES.items()):
        if col.button(label, use_container_width=True):
            (st.session_state["pickup_place"], st.session_state["dropoff_place"],
             date_str, time_str, st.session_state["passenger_count"]) = vals
            st.session_state["date_val"] = pd.to_datetime(date_str).date()
            st.session_state["time_val"] = pd.to_datetime(time_str).time()
            st.rerun()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Pickup")
    pickup_place = st.selectbox("Pickup location", list(LANDMARKS.keys()), key="pickup_place")
    pu_lon = st.number_input("Pickup longitude", value=st.session_state["pu_lon"], format="%.4f", key="pu_lon")
    pu_lat = st.number_input("Pickup latitude", value=st.session_state["pu_lat"], format="%.4f", key="pu_lat")

with col2:
    st.subheader("Drop-off")
    dropoff_place = st.selectbox("Drop-off location", list(LANDMARKS.keys()), key="dropoff_place")
    do_lon = st.number_input("Drop-off longitude", value=st.session_state["do_lon"], format="%.4f", key="do_lon")
    do_lat = st.number_input("Drop-off latitude", value=st.session_state["do_lat"], format="%.4f", key="do_lat")

col3, col4, col5 = st.columns(3)
with col3:
    date_val = st.date_input("Date", key="date_val")
with col4:
    time_val = st.time_input("Time", key="time_val")
with col5:
    passenger_count = st.slider("Passengers", 1, 6, key="passenger_count")

if st.button("Estimate Fare", type="primary", use_container_width=True):
    try:
        fare, distance, timestamp, rush, weekend = predict_fare(
            pickup_place, dropoff_place, pu_lon, pu_lat, do_lon, do_lat,
            date_val, time_val, passenger_count,
        )

        st.success(f"### Estimated Taxi Fare: ${fare:.2f}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Trip Distance", f"{distance:.2f} km")
        m2.metric("Rush hour", "Yes" if rush else "No")
        m3.metric("Weekend", "Yes" if weekend else "No")
        st.write(f"**Pickup time:** {timestamp.strftime('%A, %d %b %Y at %H:%M')}")
        st.write(f"**Passengers:** {int(passenger_count)}")
        st.caption(
            "Predicted by a deep feedforward neural network trained on the "
            "NYC Taxi Fare dataset (2009–2015 fares)."
        )
    except Exception as e:
        st.error(f"Error: {e}")
