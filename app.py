import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# === ML Tools ===
from haversine import haversine, Unit

# === Load Model ===
model = joblib.load("model.pkl")

# === Style ===
st.set_page_config(page_title="Delivery Delay Predictor", layout="wide")
st.title("🚚 Delivery Performance Dashboard")

# === Tabs ===
tab1, tab2 = st.tabs(["🚚 Delay Predictor", "📊 Insights Dashboard"])

# === Load Dataset ===
df = pd.read_excel("delivery_data.xlsx")

# === Ensure Required Columns Exist ===
if 'Dispatch Delay (days)' not in df.columns or 'Delivery Time (days)' not in df.columns or 'Delayed?' not in df.columns:
    if 'Order Date' in df.columns and 'Dispatch Date' in df.columns and 'Delivery Date' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Dispatch Date'] = pd.to_datetime(df['Dispatch Date'])
        df['Delivery Date'] = pd.to_datetime(df['Delivery Date'])

        df['Dispatch Delay (days)'] = (df['Dispatch Date'] - df['Order Date']).dt.days
        df['Delivery Time (days)'] = (df['Delivery Date'] - df['Order Date']).dt.days
        df['Delayed?'] = df['Delivery Time (days)'].apply(lambda x: 'Yes' if x > 3 else 'No')
    else:
        st.error("❌ Required date columns are missing in your Excel sheet.")
        st.stop()

with tab1:
    st.header("📦 Predict Delivery Delay")

    col1, col2 = st.columns(2)

    with col1:
        region = st.selectbox("Region", ['East', 'West', 'North', 'South'])
        distance = st.slider("Distance (km)", 0, 1000, 200)

    with col2:
        weather = st.selectbox("Weather", ['Clear', 'Clouds', 'Rain', 'Storm', 'Fog'])
        dispatch_delay = st.slider("Dispatch Delay (days)", 0, 5, 1)

    # Encode for prediction
    region_map = {'East': 0, 'West': 1, 'North': 2, 'South': 3}
    weather_map = {'Clear': 0, 'Clouds': 1, 'Rain': 2, 'Storm': 3, 'Fog': 4}

    region_code = region_map[region]
    weather_code = weather_map[weather]

    sample = pd.DataFrame({
        'Region': [region_code],
        'Weather': [weather_code],
        'Distance (km)': [distance],
        'Dispatch Delay (days)': [dispatch_delay]
    })

    if st.button("🔍 Predict Delay"):
        prediction = model.predict(sample)[0]
        st.success("Prediction: **DELAYED** 🚨" if prediction == 1 else "Prediction: **On Time** ✅")

        # Add prediction to dataframe for dashboard
        new_row = {
            'Region': region,
            'Weather': weather,
            'Distance (km)': distance,
            'Dispatch Delay (days)': dispatch_delay,
            'Delivery Time (days)': 3 + dispatch_delay + (1 if prediction == 1 else 0),
            'Delayed?': 'Yes' if prediction == 1 else 'No'
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

with tab2:
    st.header("📈 Delivery Insights")

    # KPI Metrics
    total_orders = len(df)
    delayed_orders = df['Delayed?'].value_counts().get('Yes', 0)
    delay_percentage = round((df['Delayed?'] == 'Yes').mean() * 100, 2)
    worst_region = df.groupby("Region")['Delivery Time (days)'].mean().idxmax()
    worst_weather = df.groupby("Weather")['Delivery Time (days)'].mean().idxmax()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Orders", total_orders)
    kpi2.metric("Delayed Deliveries", delayed_orders)
    kpi3.metric("Delay %", f"{delay_percentage}%")
    kpi4.metric("Worst Region", worst_region)

    # === Charts ===
    with st.expander("📍 Average Delivery Time by Region"):
        fig, ax = plt.subplots()
        region_avg = df.groupby('Region')['Delivery Time (days)'].mean()
        sns.barplot(x=region_avg.index, y=region_avg.values, hue=region_avg.index, palette="coolwarm", legend=False, ax=ax)
        ax.set_title("Avg Delivery Time by Region")
        st.pyplot(fig)

    with st.expander("⛈️ Weather Impact on Delivery"):
        fig, ax = plt.subplots()
        weather_avg = df.groupby('Weather')['Delivery Time (days)'].mean()
        sns.barplot(x=weather_avg.index, y=weather_avg.values, hue=weather_avg.index, palette="BuGn_r", legend=False, ax=ax)
        ax.set_title("Avg Delivery Time by Weather")
        st.pyplot(fig)

    with st.expander("📏 Distance vs Delivery Time"):
        fig, ax = plt.subplots()
        ax.scatter(df['Distance (km)'], df['Delivery Time (days)'], alpha=0.6, color='orange')
        ax.set_title("Distance vs Delivery Time")
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Delivery Time (days)")
        st.pyplot(fig)

    with st.expander("🧪 Correlation Heatmap"):
        fig, ax = plt.subplots()
        corr = df[['Delivery Time (days)', 'Dispatch Delay (days)', 'Distance (km)']].corr()
        sns.heatmap(corr, annot=True, cmap='Blues', ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
