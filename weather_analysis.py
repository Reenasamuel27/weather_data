import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

# Coustom for Sidebar Background
st.set_page_config(
    page_title="Custom Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for sidebar background color
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            background: url("https://img.freepik.com/premium-photo/blue-white-abstract-background-with-sun-shining-water_1097558-5275.jpg") no-repeat center center fixed;
            background-size: cover;
            color: #343434 !important;  /* White Text */
        }
        section[data-testid="stSidebar"] * {
            color: #343434 !important;  /* Ensures all text inside is white */
        }
    </style>
    """,
    unsafe_allow_html=True
)

## Coustom for Visual side Background
st.markdown(
    """
    <style>
        .stApp {
            background: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTiWRYZfJUlluExMWJIu_lcLRDfUnH08p5R5w&s") no-repeat center center fixed;
            background-size: cover;
        }

    </style>
    """,
    unsafe_allow_html=True
)

 #Read CSV data 
df = pd.read_csv("large_weather_data.csv")
#Covert the date time and month format
df["Datetime"] = pd.to_datetime(df["Datetime"])
df["Month"] = df["Datetime"].dt.month_name()


# 🔄Sidebar Refresh Data
if st.sidebar.button("🔄 Refresh"):
    st.sidebar.experimental_rerun()
    st.sidebar.markdown("---")

     #Sidebar for City & Month Selection
st.sidebar.header("🌍 Select Filters")
selected_city = st.sidebar.selectbox("Choose City", df["City"].unique())
selected_month = st.sidebar.selectbox("Select Month", df["Month"].unique())
 
# 📊 Filter Data
filtered_data = df[(df["City"] == selected_city) & (df["Month"] == selected_month)]

# Load CSV file
df = pd.read_csv("large_weather_data.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df["Timestamp"] = df["Datetime"].astype("int64") // 10**9  # Convert date to timestamp
 
#Select City for Prediction in sidebar
selected_Country = st.sidebar.selectbox("📍 Choose Country", df["Country"].unique())
Country_data = df[df["Country"] == selected_Country]

#Sidebar Clear filters
if st.sidebar.button("❌ Clear filter"):
    st.sidebar.experimental_rerun()
    st.sidebar.markdown("---")

#Sidebar Download Filtered Data
st.sidebar.download_button(
    label="🔽Download file",
    data=filtered_data.to_csv(index=False),
    file_name=f"{selected_city}_{selected_month}_Weather.csv",
    mime="text/csv",
   ) 

#Dashboard Title main background
st.title(f"🌤️ Weather data Analysis ")

#Average for temp,wind speed, AQI, Humidity city and month wise

st.subheader(f"⛅ Weather Summary for {selected_city} -{selected_month}")
col1, col2= st.columns(2)
with col1:
    st.metric("🌡️ Average Temperature (°C)", round(filtered_data["Temperature_C"].mean(), 2))
with col2:
    st.metric("💨 Average Wind Speed (km/h)", round(filtered_data["WindSpeed_kmh"].mean(), 2))
col3, col4 = st.columns(2)    
with col3:
    st.metric("🏭 Average AQI", round(filtered_data["AQI"].mean(), 2))
with col4:   
    st.metric("💧 Average Humidity (%)", round(filtered_data["Humidity_%"].mean(), 2))

#Pie chart:Average for temp,wind speed, AQI, Humidity city   
# Filter Data for Selected City
city_data = df[df["City"] == selected_city]

# 📌 **Show Average Values for the Selected City**
st.subheader(f"🌆 Weather Overview for {selected_city}")

    # Prepare data for Pie Chart
pie_data = {
    "Parameter": ["Temperature_C", "WindSpeed_kmh", "AQI", "Humidity_%"],
    "Value": [
        city_data["Temperature_C"].values[0],
        city_data["WindSpeed_kmh"].values[0],
        city_data["AQI"].values[0],
        city_data["Humidity_%"].values[0],
        
    ],
}

pie_df = pd.DataFrame(pie_data)

# Create Pie Chart
fig_pie = px.pie(pie_df, names="Parameter", values="Value", title="Weather Parameter Distribution", hole=0.3)

# 🔹 Set Transparent Background
fig_pie.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
    paper_bgcolor='rgba(0,0,0,0)',  # Transparent figure background
    font=dict(color="white"),  # White text for better visibility
)
st.plotly_chart(fig_pie, use_container_width=True)
# Footer
st.markdown("🔹 Data is for demonstration purposes only.")

#Pie Chart:Create Weather Condition Column
if "Weather_Condition" not in df.columns:
    def classify_weather(row):
     if row["Precipitation_mm"] > 5:
        return "Rainy"
     elif row["Temperature_C"] > 30 and row["Humidity_%"] < 50:
        return "Sunny"
     elif row["WindSpeed_kmh"] > 30:
        return "Windy"
     elif row["AQI"] > 150:
      return "Poor Air Quality"
     else:
      return "Cloudy"
 
df["Weather_Condition"] = df.apply(classify_weather, axis=1)
 
# Updating new coloumn
df["Month"] = df["Datetime"].dt.month_name()
filtered_data = df[(df["City"] == selected_city) & (df["Month"] == selected_month)]
 
# Weather Condition Pie Chart (Fixed)
city_data = df[df["City"] == selected_city]
st.subheader("☁️ Weather Condition Distribution")
fig_pie = px.pie(
    filtered_data,
    names="Weather_Condition",
    title="Weather Breakdown",
    color_discrete_sequence=px.colors.sequential.Plasma
)
    # 🔹 Set Transparent Background
fig_pie.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
    paper_bgcolor='rgba(0,0,0,0)',  # Transparent figure background
    font=dict(color="white"),  # White text for better visibility
)

st.plotly_chart(fig_pie, use_container_width=True, key="weather_pie_chart")

# Footer
st.markdown("🔹 Data is for demonstration purposes only.")

# 📌 Display Subheader
st.subheader("🔥 Monthly Solar Radiation Intensity")
df["Month"] = df["Datetime"].dt.month_name()
monthly_avg_solar = df[df["City"] == selected_city].groupby("Month")["SolarRadiation_Wm2"].mean().reset_index()
# 📊 Create Bar Chart with Heat-Based Color Coding
fig_solar = px.bar(
        df,
        x="Month",
        y="SolarRadiation_Wm2",
        color="SolarRadiation_Wm2",  # Use Solar Radiation for color
        title="☀️ Solar Radiation (W/m²) - Heat Intensity by Month",
        labels={"SolarRadiation_Wm2": "Solar Radiation (W/m²)"},
        color_continuous_scale="YlOrRd",  # Yellow → Orange → Red (Heatmap effect)
)
 # 🔹 Set Transparent Background
fig_solar.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
    paper_bgcolor='rgba(0,0,0,0)',  # Transparent figure background
    font=dict(color="white"),  # White text for better visibility
)
st.plotly_chart(fig_solar, use_container_width=True)


# 🔹 Footer
st.markdown("🔹 Data is for demonstration purposes only.")

#import library function
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

#Monthly Temperature Trends
st.subheader("📈 Monthly Temperature Trends")
df["Month"] = df["Datetime"].dt.month_name()
monthly_avg_temp = df[df["City"] == selected_city].groupby("Month")["Temperature_C"].mean().reset_index()
fig_monthly_temp = px.bar(
    monthly_avg_temp, x="Month", y="Temperature_C",
    title="🌡️ Average Temperature by Month",
    color="Temperature_C",
    color_continuous_scale="Viridis"
)

# 🔹 Set Transparent Background
fig_monthly_temp.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
    paper_bgcolor='rgba(0,0,0,0)',  # Transparent figure background
    font=dict(color="white"),  # White text for better visibility
)
st.plotly_chart(fig_monthly_temp, key="temp_trend_chart")

#AI based report
st.subheader("🔮 AI-Based Weather Forecast")
#Train ML Model
X = Country_data["Timestamp"].values.reshape(-1, 1)
y = Country_data["Temperature_C"]
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 
# 🔮Predict Next 7 Days
future_dates = [(Country_data["Timestamp"].max() + i * 86400) for i in range(1, 8)]
future_temps = model.predict(np.array(future_dates).reshape(-1, 1))
 
forecast_df = pd.DataFrame({
    "Date": pd.to_datetime(future_dates, unit="s"),
    "Predicted Temperature (°C)": future_temps
})
 
#Display Forecast Data
st.write(forecast_df)
st.subheader("📈 7-Day Temperature Forecast") 
# 📈Plot Forecast Graph
fig_forecast = px.line(forecast_df, x="Date", y="Predicted Temperature (°C)")
fig_forecast.update_traces(line=dict(color='rgba(0,0,255,0.9)'))
fig_forecast.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig_forecast)
 
# 🚨 AI-Powered Weather Alerts
st.subheader("⚠️ Weather Alerts")
 
def weather_alerts(temp, wind, aqi):
    alerts = []
    if temp > 35:
        alerts.append("🔥 Extreme Heat Warning!")
    if wind > 40:
        alerts.append("💨 High Wind Alert!")
    if aqi > 150:
        alerts.append("🏭 Poor Air Quality! Limit Outdoor Activity.")
   
    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("✅ Weather conditions are normal.")
 
# Apply AI alerts for selected city
latest_weather = Country_data.iloc[-1]  # Latest recorded weather
weather_alerts(latest_weather["Temperature_C"], latest_weather["WindSpeed_kmh"], latest_weather["AQI"])

#print overall file data
st.header("Overall Summary data")
if 'filtered_data' in locals():
    st.write(filtered_data.head())
else:
    st.error("🚨 filtered_data is not defined. Check your filtering logic!")
filtered_data.head()