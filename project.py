import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from haversine import haversine, Unit
import joblib

# === Seaborn Style ===
sns.set(style="whitegrid")

# === Load Excel File ===
file_path = 'delivery_data.xlsx'
df = pd.read_excel(file_path)

# === Date Handling ===
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Dispatch Date'] = pd.to_datetime(df['Dispatch Date'])
df['Delivery Date'] = pd.to_datetime(df['Delivery Date'])

# === Calculate Delivery Time, Delays ===
df['Delivery Time (days)'] = (df['Delivery Date'] - df['Order Date']).dt.days
df['Dispatch Delay (days)'] = (df['Dispatch Date'] - df['Order Date']).dt.days
df['Delayed?'] = df['Delivery Time (days)'].apply(lambda x: 'Yes' if x > 3 else 'No')

# === Save Processed Data ===
df.to_excel('delivery_analysis_output.xlsx', index=False)

# === KPI Summary ===
total_orders = len(df)
delayed_orders = df['Delayed?'].value_counts().get('Yes', 0)
delay_percentage = round((df['Delayed?'] == 'Yes').mean() * 100, 2)
worst_region = df.groupby("Region")['Delivery Time (days)'].mean().idxmax()
worst_weather = df.groupby("Weather")['Delivery Time (days)'].mean().idxmax()

print("\n📊 KPI SUMMARY")
print("📦 Total Orders:", total_orders)
print("⏱️ Delayed Deliveries:", delayed_orders)
print("⚠️ Delay Percentage:", delay_percentage, "%")
print("📍 Worst Region:", worst_region)
print("⛈️ Worst Weather Condition:", worst_weather)

# === Plot 1: Avg Delivery Time by Region ===
plt.figure(figsize=(8, 5))
region_avg = df.groupby('Region')['Delivery Time (days)'].mean()
sns.barplot(
    x=region_avg.index,
    y=region_avg.values,
    hue=region_avg.index,
    palette='coolwarm',
    legend=False
).set_title('Average Delivery Time by Region')
plt.ylabel('Days')
for i, v in enumerate(region_avg.values):
    plt.text(i, v + 0.2, f"{v:.1f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('region_avg_delivery.png')
plt.show()

# === Plot 2: Avg Delivery Time by Weather ===
plt.figure(figsize=(8, 5))
weather_avg = df.groupby('Weather')['Delivery Time (days)'].mean()
sns.barplot(
    x=weather_avg.index,
    y=weather_avg.values,
    hue=weather_avg.index,
    palette='BuGn_r',
    legend=False
).set_title('Average Delivery Time by Weather')
plt.ylabel('Days')
for i, v in enumerate(weather_avg.values):
    plt.text(i, v + 0.2, f"{v:.1f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('weather_impact.png')
plt.show()

# === Plot 3: Distance vs Delivery Time ===
plt.figure(figsize=(8, 5))
plt.scatter(df['Distance (km)'], df['Delivery Time (days)'], alpha=0.7, color='orange')
plt.title('Distance vs Delivery Time')
plt.xlabel('Distance (km)')
plt.ylabel('Delivery Time (days)')
plt.grid(True)
plt.tight_layout()
plt.savefig('distance_vs_delivery.png')
plt.show()

# === Plot 4: Correlation Heatmap ===
plt.figure(figsize=(6, 4))
corr = df[['Delivery Time (days)', 'Dispatch Delay (days)', 'Distance (km)']].corr()
sns.heatmap(corr, annot=True, cmap='Blues')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

# === Encode for ML ===
df_encoded = df.copy()
df_encoded['Region'] = df_encoded['Region'].astype('category').cat.codes
df_encoded['Weather'] = df_encoded['Weather'].astype('category').cat.codes
df_encoded['Delayed?'] = df_encoded['Delayed?'].map({'No': 0, 'Yes': 1})

# === ML Training ===
X = df_encoded[['Region', 'Weather', 'Distance (km)', 'Dispatch Delay (days)']]
y = df_encoded['Delayed?']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n📈 MACHINE LEARNING PREDICTION REPORT:")
print(classification_report(y_test, y_pred))

# === MOCK WEATHER + DISTANCE SETUP ===
mock_weather = {
    'Chennai': 'Rain',
    'Mumbai': 'Storm',
    'Delhi': 'Clear',
    'Bangalore': 'Clouds',
    'Kolkata': 'Fog'
}
weather_map = {'Clear': 0, 'Clouds': 1, 'Rain': 2, 'Storm': 3, 'Fog': 4}
region_map = {'East': 0, 'West': 1, 'North': 2, 'South': 3}

def get_mock_weather(city):
    return mock_weather.get(city, 'Clear')

def calculate_distance_km(origin_coords, dest_coords):
    return round(haversine(origin_coords, dest_coords, unit=Unit.KILOMETERS), 2)

# === Test: Chennai to Mumbai ===
origin_city = 'Chennai'
destination_city = 'Mumbai'
origin_coords = (13.0827, 80.2707)
destination_coords = (19.0760, 72.8777)

weather = get_mock_weather(destination_city)
distance_km = calculate_distance_km(origin_coords, destination_coords)

weather_code = weather_map.get(weather, 0)
region_code = 1  # Example: West

sample = pd.DataFrame({
    'Region': [region_code],
    'Weather': [weather_code],
    'Distance (km)': [distance_km],
    'Dispatch Delay (days)': [2]
})

prediction = model.predict(sample)[0]
print("\n🌦️ Weather in", destination_city + ":", weather)
print("📏 Distance:", distance_km, "km")
print("🚚 Predicted Delay:", "Yes" if prediction == 1 else "No")

# === 💾 SAVE MODEL ===
joblib.dump(model, "model.pkl")
print("\n✅ Trained model saved as model.pkl")
