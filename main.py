import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RetailSalesForecaster:
    def __init__(self):
        self.data = None
        self.forecast_results = {}
    
    def load_sample_data(self):
        """Generate sample retail sales data"""
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        
        # Base sales with trend and seasonality
        base_sales = 1000
        trend = np.linspace(0, 500, len(dates))
        
        # Weekly seasonality (higher sales on weekends)
        weekly_pattern = np.sin(2 * np.pi * np.arange(len(dates)) / 7) * 200
        
        # Monthly seasonality (holiday effects)
        monthly_pattern = np.sin(2 * np.pi * np.arange(len(dates)) / 365.25 * 12) * 300
        
        # Random noise
        noise = np.random.normal(0, 100, len(dates))
        
        sales = base_sales + trend + weekly_pattern + monthly_pattern + noise
        sales = np.maximum(sales, 50)  # Ensure no negative sales
        
        self.data = pd.DataFrame({
            'date': dates,
            'sales': sales
        })
        
        print("Sample data loaded successfully!")
        return self.data
    
    def load_custom_data(self, file_path):
        """Load custom sales data from CSV"""
        try:
            self.data = pd.read_csv(file_path)
            self.data['date'] = pd.to_datetime(self.data['date'])
            print(f"Custom data loaded: {len(self.data)} records")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def moving_average_forecast(self, window=30, forecast_days=30):
        """Simple moving average forecast"""
        if self.data is None:
            print("No data loaded!")
            return None
        
        # Calculate moving average
        ma = self.data['sales'].rolling(window=window).mean()
        
        # Use last moving average value for forecast
        last_ma = ma.iloc[-1]
        forecast_dates = pd.date_range(
            start=self.data['date'].iloc[-1] + timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        forecast = [last_ma] * forecast_days
        
        self.forecast_results['moving_average'] = {
            'dates': forecast_dates,
            'forecast': forecast,
            'method': f'{window}-day Moving Average'
        }
        
        return forecast_dates, forecast
    
    def exponential_smoothing_forecast(self, alpha=0.3, forecast_days=30):
        """Exponential smoothing forecast"""
        if self.data is None:
            print("No data loaded!")
            return None
        
        sales = self.data['sales'].values
        
        # Calculate exponential smoothing
        smoothed = [sales[0]]
        for i in range(1, len(sales)):
            smoothed.append(alpha * sales[i] + (1 - alpha) * smoothed[i-1])
        
        # Forecast using last smoothed value
        last_smoothed = smoothed[-1]
        forecast_dates = pd.date_range(
            start=self.data['date'].iloc[-1] + timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        forecast = [last_smoothed] * forecast_days
        
        self.forecast_results['exponential_smoothing'] = {
            'dates': forecast_dates,
            'forecast': forecast,
            'method': f'Exponential Smoothing (α={alpha})'
        }
        
        return forecast_dates, forecast
    
    def linear_trend_forecast(self, forecast_days=30):
        """Linear trend forecast"""
        if self.data is None:
            print("No data loaded!")
            return None
        
        # Create numerical date values for regression
        self.data['date_num'] = (self.data['date'] - self.data['date'].min()).dt.days
        
        # Fit linear trend
        x = self.data['date_num'].values
        y = self.data['sales'].values
        
        slope = np.cov(x, y)[0,1] / np.var(x)
        intercept = np.mean(y) - slope * np.mean(x)
        
        # Generate forecast
        last_date_num = x[-1]
        forecast_date_nums = np.arange(last_date_num + 1, last_date_num + 1 + forecast_days)
        forecast = slope * forecast_date_nums + intercept
        
        forecast_dates = pd.date_range(
            start=self.data['date'].iloc[-1] + timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        self.forecast_results['linear_trend'] = {
            'dates': forecast_dates,
            'forecast': forecast,
            'method': 'Linear Trend'
        }
        
        return forecast_dates, forecast
    
    def seasonal_naive_forecast(self, season_length=7, forecast_days=30):
        """Seasonal naive forecast (repeats seasonal pattern)"""
        if self.data is None:
            print("No data loaded!")
            return None
        
        # Get last seasonal cycle
        last_season = self.data['sales'].iloc[-season_length:].values
        
        # Repeat pattern for forecast period
        forecast = []
        for i in range(forecast_days):
            forecast.append(last_season[i % season_length])
        
        forecast_dates = pd.date_range(
            start=self.data['date'].iloc[-1] + timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        self.forecast_results['seasonal_naive'] = {
            'dates': forecast_dates,
            'forecast': forecast,
            'method': f'Seasonal Naive ({season_length}-day cycle)'
        }
        
        return forecast_dates, forecast
    
    def plot_forecasts(self, show_last_days=60):
        """Plot historical data and all forecasts"""
        if not self.forecast_results:
            print("No forecasts available! Run forecasting methods first.")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Plot historical data (last N days)
        recent_data = self.data.iloc[-show_last_days:]
        plt.plot(recent_data['date'], recent_data['sales'], 
                label='Historical Sales', linewidth=2, color='black')
        
        # Plot each forecast
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (method, result) in enumerate(self.forecast_results.items()):
            plt.plot(result['dates'], result['forecast'], 
                    label=result['method'], linewidth=2, 
                    color=colors[i % len(colors)], linestyle='--')
        
        plt.title('Sales Forecasting - Multiple Methods', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Sales', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def get_forecast_summary(self):
        """Get summary of all forecasts"""
        if not self.forecast_results:
            print("No forecasts available!")
            return
        
        print("\n" + "="*60)
        print("SALES FORECAST SUMMARY")
        print("="*60)
        
        for method, result in self.forecast_results.items():
            forecast_avg = np.mean(result['forecast'])
            forecast_total = np.sum(result['forecast'])
            
            print(f"\n{result['method']}:")
            print(f"  Average daily sales: ${forecast_avg:,.2f}")
            print(f"  Total forecast period: ${forecast_total:,.2f}")
            print(f"  Forecast period: {len(result['forecast'])} days")
    
    def calculate_forecast_accuracy(self, actual_data, method='moving_average'):
        """Calculate forecast accuracy (if you have actual future data)"""
        if method not in self.forecast_results:
            print(f"Method {method} not found!")
            return None
        
        forecast = self.forecast_results[method]['forecast']
        n = min(len(actual_data), len(forecast))
        
        actual = actual_data[:n]
        pred = forecast[:n]
        
        mae = np.mean(np.abs(actual - pred))
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        rmse = np.sqrt(np.mean((actual - pred)**2))
        
        print(f"\nAccuracy for {self.forecast_results[method]['method']}:")
        print(f"  Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"  Root Mean Square Error (RMSE): ${rmse:.2f}")
        
        return {'MAE': mae, 'MAPE': mape, 'RMSE': rmse}

# Example usage
if __name__ == "__main__":
    # Create forecaster
    forecaster = RetailSalesForecaster()
    
    # Load sample data
    print("Loading sample retail sales data...")
    data = forecaster.load_sample_data()
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"Sales range: ${data['sales'].min():.2f} to ${data['sales'].max():.2f}")
    
    # Run different forecasting methods
    print("\nRunning forecasts...")
    
    # 30-day forecasts using different methods
    forecaster.moving_average_forecast(window=30, forecast_days=30)
    forecaster.exponential_smoothing_forecast(alpha=0.3, forecast_days=30)
    forecaster.linear_trend_forecast(forecast_days=30)
    forecaster.seasonal_naive_forecast(season_length=7, forecast_days=30)
    
    # Display results
    forecaster.get_forecast_summary()
    
    # Plot results
    print("\nGenerating forecast plot...")
    forecaster.plot_forecasts(show_last_days=90)
    
    # Example of loading custom data (commented out)
    # forecaster.load_custom_data('your_sales_data.csv')
    # CSV should have columns: 'date' and 'sales'