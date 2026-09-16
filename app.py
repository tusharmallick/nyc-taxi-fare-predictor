"""
NYC Taxi Fare Prediction — deployment app
Lab Assignment 02 | Deep Feedforward Neural Network

Run locally:   python app.py
On HF Spaces:  this file is the entrypoint, nothing else needed.
"""

import os
import numpy as np
import pandas as pd
import joblib
import gradio as gr
from tensorflow import keras

# ---------------------------------------------------------------------------
# Load artifacts (same folder as this file)
# ---------------------------------------------------------------------------
MODEL_PATH = "final_taxi_fare_model.keras"
SCALER_PATH = "scaler.pkl"

model = keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

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


def predict_fare(pickup_place, dropoff_place,
                 pu_lon, pu_lat, do_lon, do_lat,
                 date_str, time_str, passenger_count):
    try:
        # Resolve landmark presets, falling back to the manual coordinate boxes
        if LANDMARKS.get(pickup_place):
            pu_lon, pu_lat = LANDMARKS[pickup_place]
        if LANDMARKS.get(dropoff_place):
            do_lon, do_lat = LANDMARKS[dropoff_place]

        timestamp = pd.to_datetime(f"{date_str} {time_str}")

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

        rush = "Yes" if row["is_rush_hour"].iloc[0] else "No"
        weekend = "Yes" if row["is_weekend"].iloc[0] else "No"

        result = f"## Estimated Taxi Fare: ${fare:.2f}\n\n"
        result += f"**Trip Distance:** {distance:.2f} km\n\n"
        result += f"**Pickup time:** {timestamp.strftime('%A, %d %b %Y at %H:%M')}\n\n"
        result += f"**Rush hour:** {rush} &nbsp;&nbsp;|&nbsp;&nbsp; **Weekend:** {weekend}\n\n"
        result += f"**Passengers:** {int(passenger_count)}\n\n"
        result += "---\n*Predicted by a deep feedforward neural network trained on "
        result += "the NYC Taxi Fare dataset (2009–2015 fares).*"
        return result

    except Exception as e:
        return f"**Error:** {e}\n\nCheck that the date is YYYY-MM-DD and the time is HH:MM."


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
with gr.Blocks(title="NYC Taxi Fare Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # NYC Taxi Fare Predictor
        Deep Feedforward Neural Network | Lab Assignment 02

        Pick a landmark or enter coordinates manually, set the trip time, and get an
        estimated fare.
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Pickup")
            pickup_place = gr.Dropdown(
                choices=list(LANDMARKS.keys()),
                value="Empire State Building",
                label="Pickup location",
            )
            pu_lon = gr.Number(value=-73.9857, label="Pickup longitude")
            pu_lat = gr.Number(value=40.7484, label="Pickup latitude")

        with gr.Column():
            gr.Markdown("### Drop-off")
            dropoff_place = gr.Dropdown(
                choices=list(LANDMARKS.keys()),
                value="JFK Airport",
                label="Drop-off location",
            )
            do_lon = gr.Number(value=-73.7781, label="Drop-off longitude")
            do_lat = gr.Number(value=40.6413, label="Drop-off latitude")

    with gr.Row():
        date_str = gr.Textbox(value="2015-06-15", label="Date (YYYY-MM-DD)")
        time_str = gr.Textbox(value="18:30", label="Time (HH:MM, 24-hour)")
        passenger_count = gr.Slider(1, 6, value=1, step=1, label="Passengers")

    predict_btn = gr.Button("Estimate Fare", variant="primary", size="lg")
    output = gr.Markdown()

    predict_btn.click(
        fn=predict_fare,
        inputs=[pickup_place, dropoff_place, pu_lon, pu_lat, do_lon, do_lat,
                date_str, time_str, passenger_count],
        outputs=output,
    )

    gr.Examples(
        examples=[
            ["Empire State Building", "JFK Airport", -73.9857, 40.7484, -73.7781, 40.6413, "2015-06-15", "18:30", 1],
            ["Times Square", "Central Park", -73.9855, 40.7580, -73.9654, 40.7829, "2015-03-10", "09:00", 2],
            ["Wall Street", "LaGuardia Airport", -74.0089, 40.7061, -73.8740, 40.7769, "2015-11-21", "14:15", 3],
        ],
        inputs=[pickup_place, dropoff_place, pu_lon, pu_lat, do_lon, do_lat,
                date_str, time_str, passenger_count],
    )


if __name__ == "__main__":
    demo.launch()
