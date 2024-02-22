import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)
    return data

def calculate_sub_index(value, pollutant):
    if pollutant == 'PM2.5':
        if value <= 30:
            return value * (50 / 30)
        elif 30 < value <= 60:
            return 50 + (value - 30) * (50 / 30)
        elif 60 < value <= 90:
            return 100 + (value - 60) * (100 / 30)
        elif 90 < value <= 120:
            return 200 + (value - 90) * (100 / 30)
        elif 120 < value <= 250:
            return 300 + (value - 120) * (100 / 130)
        elif value > 250:
            return 400 + (value - 250) * (100 / 130)

    elif pollutant == 'PM10':
        if value <= 50:
            return value
        elif 50 < value <= 100:
            return value
        elif 100 < value <= 250:
            return 100 + (value - 100) * (100 / 150)
        elif 250 < value <= 350:
            return 200 + (value - 250)
        elif 350 < value <= 430:
            return 300 + (value - 350) * (100 / 80)
        elif value > 430:
            return 400 + (value - 430) * (100 / 80)
        
    elif pollutant == 'NO2':
        if value <= 40:
            return value * (50 / 40)
        elif 40 < value <= 80:
            return 50 + (value - 40) * (50 / 40)
        elif 80 < value <= 180:
            return 100 + (value - 80) * 100 / 100
        elif 180 < value <= 280:
            return 200 + (value - 180) * (100 / 100)
        elif 280 < value <= 400:
            return 300 + (value - 280) * (100 / 120)
        elif value > 400:
            return 400 + (value - 400) * (100 / 120)
   
    elif pollutant == 'SO2':
        if value <= 40:
            return value * (50 / 40)
        elif 40 < value <= 80:
            return 50 + (value - 40) * (50 / 40)
        elif 80 < value <= 380:
            return 100 + (value - 80) * (100 / 300)
        elif 380 < value <= 800:
            return 200 + (value - 380) * (100 / 420)
        elif 800 < value <= 1600:
            return 300 + (value - 800) * (100 / 800)
        elif value > 1600:
            return 400 + (value - 1600) * (100 / 800)

        
    elif pollutant == 'CO':
        if value <= 1:
            return value * (50 / 1)
        elif 1 < value <= 2:
            return 50 + (value - 1) * (50 / 1)
        elif 2 < value <= 10:
            return 100 + (value - 2) * (100 / 8)
        elif 10 < value <= 17:
            return 200 + (value - 10) * (100 / 7)
        elif 17 < value <= 34:
            return 300 + (value - 17) * (100 / 17)
        elif value > 34:
            return 400 + (value - 34) * (100 / 17)

    elif pollutant == 'Ozone':
        if value <= 50:
            return value * 50 / 50
        elif 50 < value <= 100:
            return 50 + (value - 50) * 50 / 50
        elif 100 < value <= 168:
            return 100 + (value - 100) * 100 / 68
        elif 168 < value <= 208:
            return 200 + (value - 168) * (100 / 40)
        elif 208 < value <= 748:
            return 300 + (value - 208) * (100 / 539)
        elif value > 748:
            return 400 + (value - 400) * (100 / 539)

    else:
        return None

def calculate_aqi(data):
    pollutants = ['PM2.5', 'PM10', 'Ozone', 'NO2', 'SO2', 'CO']
    for pollutant in pollutants:
        data[pollutant + '_Sub_Index'] = data[pollutant].apply(lambda x: calculate_sub_index(x, pollutant))
    data['AQI'] = data[[f'{p}_Sub_Index' for p in pollutants]].max(axis=1)
    return data

def fit_prophet_model(data):
    prophet_data = data[['Date', 'AQI']].rename(columns={'Date':'ds', 'AQI':'y'})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(prophet_data)
    return model

def get_user_input():
    start_date = input("Enter the start date after 2023-08-31 (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")
    return start_date, end_date

def main():
    # Load Data
    data = load_data('environmental_data.csv')
    
    # Calculate AQI
    data = calculate_aqi(data)

    # Fit Prophet Model
    model = fit_prophet_model(data)

    # Get User Input
    start_date, end_date = get_user_input()

    # Create Future Dates
    future = model.make_future_dataframe(periods=365)  # Adjust periods as needed

    # Filter Future Dates based on User Input
    future = future[(future['ds'] >= start_date) & (future['ds'] <= end_date)]

    # Predict
    forecast = model.predict(future)

    # Save Forecasted Data to CSV
    forecast.to_csv('forecasted_AQI.csv', index=False)

    # Plot Data
    plt.figure(figsize=(10, 5))
    plt.plot(data['Date'], data['AQI'], label='Historical AQI', color='blue')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecasted AQI', color='red')
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.title('Historical vs Forecasted AQI')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
