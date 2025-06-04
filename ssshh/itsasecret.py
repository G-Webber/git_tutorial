
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import yfinance as yf

def generate_synthetic_data(days: int = 365, initial_price: float = 50000) -> pd.DataFrame:
    """Generate synthetic historical crypto price data.
    
    Args:
        days (int): The number of days for which to generate data.
        initial_price (float): The starting price for the synthetic data.
        
    Returns:
        pd.DataFrame: A dataframe with columns = ['Date', 'Price'].
    """
    
    # Generate date range
    date_range = pd.date_range(end=datetime.today(), periods=days).tolist()
    
    # Generate synthetic price data
    price_data = [initial_price]
    for _ in range(1, days):
        price_data.append(price_data[-1] * (1 + np.random.normal(0, 0.01))) # Simulate daily returns with a small random walk
    
    # Create DataFrame
    df = pd.DataFrame({'Date': date_range, 'Price': price_data})
    
    return df

def get_real_crypto_data(ticker: str = "BTC-USD", period: str = '1y') -> pd.DataFrame:
    """Fetch historical crypto price data using yfinance. 
    if there are multi-level columens flatten the column names to a single level with the column headings 'Date', 'Price'.
    Return a dataframe with columns = ['Date', 'Price'] """
    try:
        # Fetch historical market data
        crypto_data = yf.Ticker(ticker)
        hist_data = crypto_data.history(period=period)

        if hist_data.empty:
            print(f"No data found for ticker {ticker} and period {period}.")
            return pd.DataFrame(columns=['Date', 'Price'])

        # Reset index to make 'Date' a column
        # yfinance typically returns the date as the index
        hist_data = hist_data.reset_index()

        # Select and rename columns
        # We are interested in the 'Close' price and the 'Date'
        # The date column might be named 'Date' or 'Datetime' depending on yfinance version/settings
        
        # Identify the correct date column (case-insensitive check)
        date_col_name = None
        if 'Date' in hist_data.columns:
            date_col_name = 'Date'
        elif 'Datetime' in hist_data.columns: # yfinance sometimes uses 'Datetime'
            date_col_name = 'Datetime'
        else:
            # Try to find a date-like column by checking for common names
            possible_date_cols = [col for col in hist_data.columns if isinstance(col, str) and 'date' in col.lower()]
            if possible_date_cols:
                date_col_name = possible_date_cols[0]
            else:
                print("Error: Could not identify the date column in the fetched data.")
                return pd.DataFrame(columns=['Date', 'Price'])
        
        # Ensure 'Close' price column exists
        if 'Close' not in hist_data.columns:
            print("Error: 'Close' price column not found in the fetched data.")
            print(f"Available columns: {hist_data.columns.tolist()}")
            return pd.DataFrame(columns=['Date', 'Price'])

        # Create the final DataFrame
        df_price = hist_data[[date_col_name, 'Close']].copy() # Use .copy() to avoid SettingWithCopyWarning
        df_price.rename(columns={date_col_name: 'Date', 'Close': 'Price'}, inplace=True)
        
        # Ensure 'Date' column is datetime type (it usually is, but good to be sure)
        df_price['Date'] = pd.to_datetime(df_price['Date'])

        return df_price

    except Exception as e:
        print(f"An error occurred while fetching or processing data for {ticker}: {e}")
        return pd.DataFrame(columns=['Date', 'Price'])

# Function to preprocess the data
def preprocess_data(data: pd.DataFrame, exchange_rate : int = 20) -> pd.DataFrame:
    """
    Date is not the index it is part of the dataframe. 
    Preprocess the data by adding a column for days since the start.
    and converting the currency by multiplying by the exchange rate the given exchange rate.
    output df With columns 'Date' 'Price' and 'Days'"""
    # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Calculate the number of days since the first date
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days

    # Convert the 'Price' column by multiplying with the exchange rate
    data['Price'] = data['Price'] * exchange_rate

    # Select the required columns
    data = data[['Date', 'Price', 'Days']]
    return data

# Function to train a linear regression model
def train_model(data: pd.DataFrame) -> LinearRegression:
    """
    Fit a sklearn Linear Regression model to the data use Days to predict Price. 
    input df With columns 'Date' 'Price' and 'Days'.
    return the trained model."""
    X = data[['Days']]  # Feature
    y = data['Price']   # Target

    model = LinearRegression()
    model.fit(X, y)
    return model

    return model


# Function to predict future prices
def predict_future_price(model: LinearRegression, days_ahead: int) -> float:
    """
    Use the Linear Regression model to predict Price on a specific day. 
    """
    # Reshape the input to match the model's expected input format
    days_ahead_array = np.array([[days_ahead]])
    
    # Predict and return the price
    predicted_price = model.predict(days_ahead_array)
    return predicted_price[0]


# Function to calculate investment value

def calculate_investment_value(initial_investment: float, predicted_price: float, current_price: float) -> float:
    """
    Calculate the future value of an investment based on predicted and current prices.

    Parameters:
    initial_investment (float): The amount of money initially invested.
    predicted_price (float): The predicted future price of the asset.
    current_price (float): The current price of the asset.

    Returns:
    float: The estimated future value of the investment.
    """
    
    if current_price == 0:
        raise ValueError("Current price cannot be zero.")
    
    return initial_investment * (predicted_price / current_price)


# Main function to run the project
def main():
    # Generate synthetic data for 365 days
    data_gen = generate_synthetic_data(365)
    data = get_real_crypto_data()
    # Preprocess the data
    data = preprocess_data(data)
    
    # Train the model
    model = train_model(data)
    
    # Predict the price 30 days into the future
    days_ahead = data['Days'].max() + 30
    predicted_price = predict_future_price(model, days_ahead)
    
    # Calculate investment value
    initial_investment = 1000.0
    current_price = data['Price'].iloc[-1]
    investment_value = calculate_investment_value(initial_investment, predicted_price, current_price)
    
    # Print results
    print(f"Predicted price after 30 days: R{predicted_price:.2f}")
    print(f"Investment value based on prediction: R{investment_value:.2f} (from an initial R{initial_investment:.2f})")
    
    # Plot the data and prediction
    plt.figure(figsize=(10, 6))
    plt.plot(data['Days'], data['Price'], label='Historical Prices')
    plt.plot([days_ahead], [predicted_price], 'ro', label='Predicted Price')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title('Cryptocurrency Price Prediction')
    plt.legend()
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()

    #input("Press Enter to continue...")
    pass
