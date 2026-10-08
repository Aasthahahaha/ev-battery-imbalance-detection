
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="EV Battery BMS",
    page_icon="🔋",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("ev_battery_imbalance_xgboost.pkl")
threshold = joblib.load("imbalance_threshold.pkl")
bms_features = joblib.load("bms_features.pkl")


# ==========================================
# TITLE
# ==========================================

st.title("🔋 AI-Based EV Battery Cell Imbalance Detection")

st.markdown(
    """
    **Machine Learning + Battery Management System (BMS)**

    This system uses an XGBoost machine-learning model to estimate
    the probability of EV battery cell imbalance using battery and
    BMS-related parameters.
    """
)

st.divider()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("🔧 Battery / BMS Parameters")

st.sidebar.subheader("Battery Condition")

state_of_charge = st.sidebar.number_input(
    "State of Charge (%)",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

state_of_health = st.sidebar.number_input(
    "State of Health (%)",
    min_value=0.0,
    max_value=100.0,
    value=85.0
)

battery_capacity_kwh = st.sidebar.number_input(
    "Battery Capacity (kWh)",
    min_value=1.0,
    value=60.0
)

remaining_capacity = st.sidebar.number_input(
    "Remaining Capacity",
    min_value=0.0,
    value=50.0
)

cycle_count = st.sidebar.number_input(
    "Cycle Count",
    min_value=0.0,
    value=500.0
)

capacity_loss_percent = st.sidebar.number_input(
    "Capacity Loss (%)",
    min_value=0.0,
    max_value=100.0,
    value=15.0
)


st.sidebar.subheader("Voltage")

cell_voltage_avg = st.sidebar.number_input(
    "Average Cell Voltage (V)",
    min_value=0.0,
    value=3.7
)

pack_voltage = st.sidebar.number_input(
    "Pack Voltage (V)",
    min_value=0.0,
    value=350.0
)


st.sidebar.subheader("Temperature")

cell_temperature_avg = st.sidebar.number_input(
    "Average Cell Temperature (°C)",
    value=30.0
)

cell_temperature_max = st.sidebar.number_input(
    "Maximum Cell Temperature (°C)",
    value=35.0
)

temperature_variance = st.sidebar.number_input(
    "Temperature Variance",
    min_value=0.0,
    value=2.0
)

average_ambient_temperature = st.sidebar.number_input(
    "Average Ambient Temperature (°C)",
    value=28.0
)

maximum_temperature = st.sidebar.number_input(
    "Maximum Temperature (°C)",
    value=38.0
)

minimum_temperature = st.sidebar.number_input(
    "Minimum Temperature (°C)",
    value=20.0
)


st.sidebar.subheader("Electrical")

internal_resistance = st.sidebar.number_input(
    "Internal Resistance",
    min_value=0.0,
    value=0.5
)

charge_efficiency = st.sidebar.number_input(
    "Charge Efficiency",
    min_value=0.0,
    max_value=1.0,
    value=0.92
)

discharge_efficiency = st.sidebar.number_input(
    "Discharge Efficiency",
    min_value=0.0,
    max_value=1.0,
    value=0.90
)


st.sidebar.subheader("BMS / Diagnostics")

BMS_warning_count = st.sidebar.number_input(
    "BMS Warning Count",
    min_value=0.0,
    value=1.0
)

abnormal_voltage_events = st.sidebar.number_input(
    "Abnormal Voltage Events",
    min_value=0.0,
    value=0.0
)

sensor_fault_count = st.sidebar.number_input(
    "Sensor Fault Count",
    min_value=0.0,
    value=0.0
)

previous_faults = st.sidebar.number_input(
    "Previous Faults",
    min_value=0.0,
    value=0.0
)

cooling_system_health = st.sidebar.number_input(
    "Cooling System Health",
    min_value=0.0,
    max_value=100.0,
    value=90.0
)


st.sidebar.subheader("Charging / Usage")

charging_cycles_last_month = st.sidebar.number_input(
    "Charging Cycles Last Month",
    min_value=0.0,
    value=20.0
)

fast_charge_ratio = st.sidebar.number_input(
    "Fast Charge Ratio",
    min_value=0.0,
    max_value=1.0,
    value=0.2
)

slow_charge_ratio = st.sidebar.number_input(
    "Slow Charge Ratio",
    min_value=0.0,
    max_value=1.0,
    value=0.5
)

average_charge_power_kw = st.sidebar.number_input(
    "Average Charge Power (kW)",
    min_value=0.0,
    value=20.0
)

charging_interruptions = st.sidebar.number_input(
    "Charging Interruptions",
    min_value=0.0,
    value=1.0
)

overcharge_events = st.sidebar.number_input(
    "Overcharge Events",
    min_value=0.0,
    value=0.0
)

odometer_km = st.sidebar.number_input(
    "Odometer (km)",
    min_value=0.0,
    value=30000.0
)


# ==========================================
# CREATE INPUT DATAFRAME
# ==========================================

input_data = {
    "state_of_charge": state_of_charge,
    "state_of_health": state_of_health,
    "battery_capacity_kwh": battery_capacity_kwh,
    "remaining_capacity": remaining_capacity,
    "cycle_count": cycle_count,
    "capacity_loss_percent": capacity_loss_percent,

    "cell_voltage_avg": cell_voltage_avg,
    "pack_voltage": pack_voltage,

    "cell_temperature_avg": cell_temperature_avg,
    "cell_temperature_max": cell_temperature_max,
    "temperature_variance": temperature_variance,
    "average_ambient_temperature": average_ambient_temperature,
    "maximum_temperature": maximum_temperature,
    "minimum_temperature": minimum_temperature,

    "internal_resistance": internal_resistance,
    "charge_efficiency": charge_efficiency,
    "discharge_efficiency": discharge_efficiency,

    "BMS_warning_count": BMS_warning_count,
    "abnormal_voltage_events": abnormal_voltage_events,
    "sensor_fault_count": sensor_fault_count,
    "previous_faults": previous_faults,
    "cooling_system_health": cooling_system_health,

    "charging_cycles_last_month": charging_cycles_last_month,
    "fast_charge_ratio": fast_charge_ratio,
    "slow_charge_ratio": slow_charge_ratio,
    "average_charge_power_kw": average_charge_power_kw,
    "charging_interruptions": charging_interruptions,
    "overcharge_events": overcharge_events,
    "odometer_km": odometer_km
}

input_df = pd.DataFrame([input_data])

# Ensure exact feature order
input_df = input_df[bms_features]


# ==========================================
# PREDICTION
# ==========================================

if st.button("🔍 Analyze Battery", use_container_width=True):

    probability = model.predict_proba(input_df)[0][1]

    prediction = int(probability >= threshold)

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Imbalance Probability",
            f"{probability * 100:.1f}%"
        )

    with col2:
        st.metric(
            "Decision Threshold",
            f"{threshold * 100:.0f}%"
        )

    with col3:
        status = "IMBALANCED" if prediction == 1 else "BALANCED"

        st.metric(
            "Battery Status",
            status
        )

    st.divider()

    if prediction == 1:

        st.error(
            "⚠️ POTENTIAL CELL IMBALANCE DETECTED"
        )

        st.warning(
            "The model estimates that the battery exhibits "
            "a probability of cell imbalance above the selected "
            "decision threshold."
        )

    else:

        st.success(
            "🟢 BATTERY APPEARS BALANCED"
        )

        st.info(
            "The model estimates that the battery is below "
            "the selected imbalance probability threshold."
        )

    st.divider()

    st.subheader("Battery Parameters")

    display_df = pd.DataFrame({
        "Parameter": [
            "State of Charge",
            "State of Health",
            "Capacity Loss",
            "Average Cell Voltage",
            "Pack Voltage",
            "Average Temperature",
            "Internal Resistance",
            "Cycle Count",
            "BMS Warning Count",
            "Abnormal Voltage Events"
        ],
        "Value": [
            f"{state_of_charge:.1f}%",
            f"{state_of_health:.1f}%",
            f"{capacity_loss_percent:.1f}%",
            f"{cell_voltage_avg:.3f} V",
            f"{pack_voltage:.1f} V",
            f"{cell_temperature_avg:.1f} °C",
            f"{internal_resistance:.3f}",
            f"{cycle_count:.0f}",
            f"{BMS_warning_count:.0f}",
            f"{abnormal_voltage_events:.0f}"
        ]
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


st.divider()

st.caption(
    "AI-Based EV Battery Cell Imbalance Detection | "
    "XGBoost + BMS Parameters"
)
