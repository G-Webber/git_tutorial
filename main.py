
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import yfinance as yf

def generate_synthetic_data(days: int = 365, initial_price: float = 50000) -> pd.DataFrame: #1
    """Generate synthetic historical crypto price data.
    
    Args:
        days (int): The number of days for which to generate data.
        initial_price (float): The starting price for the synthetic data.
        
    Returns:
        pd.DataFrame: A dataframe with columns = ['Date', 'Price'].
    """


def get_real_crypto_data(ticker: str = "BTC-USD", period: str = '1y') -> pd.DataFrame: #2
    """Fetch historical crypto price data using yfinance. 
    if there are multi-level columens flatten the column names to a single level with the column headings 'Date', 'Price'.
    Return a dataframe with columns = ['Date', 'Price'] """


# Function to preprocess the data
def preprocess_data(data: pd.DataFrame, exchange_rate : int = 20) -> pd.DataFrame: #3
    """
    Date is not the index it is part of the dataframe. 
    Preprocess the data by adding a column for days since the start.
    and converting the currency by multiplying by the exchange rate the given exchange rate.
    output df With columns 'Date' 'Price' and 'Days'"""


# Function to train a linear regression model
def train_model(data: pd.DataFrame) -> LinearRegression: #4
    """
    Fit a sklearn Linear Regression model to the data use Days to predict Price. 
    input df With columns 'Date' 'Price' and 'Days'.
    return the trained model."""



# Function to predict future prices
def predict_future_price(model: LinearRegression, days_ahead: int) -> float: #5
    """
    Use the Linear Regression model to predict Price on a specific day. 
    """

# Function to calculate investment value

def calculate_investment_value(initial_investment: float, predicted_price: float, current_price: float) -> float: #6
    """
    Calculate the future value of an investment based on predicted and current prices.

    Parameters:
    initial_investment (float): The amount of money initially invested.
    predicted_price (float): The predicted future price of the asset.
    current_price (float): The current price of the asset.

    Returns:
    float: The estimated future value of the investment.
    """

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
