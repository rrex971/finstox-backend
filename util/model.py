import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import nsepython as nse # Import nsepython
import os
from datetime import date, timedelta, datetime # Import date utilities

# Removed Marketstack API key

# Step 1: Fetch stock data from NSEPython
def fetch_data_nsepython(symbol, days_history=250):
    """Fetches historical equity data using nsepython."""
    print(f"\n📥 Fetching stock data for {symbol} from NSEPython...")

    # Calculate start and end dates
    end_dt = date.today()
    # Go back enough days to likely get ~500 trading days
    start_dt = end_dt - timedelta(days=days_history)

    # Format dates as DD-MM-YYYY for nsepython
    start_date_str = start_dt.strftime("%d-%m-%Y")
    end_date_str = end_dt.strftime("%d-%m-%Y")
    print(f"📅 Requesting data from {start_date_str} to {end_date_str}")

    try:
        # Fetch data using nsepython
        df = nse.equity_history(symbol=symbol, series="EQ", start_date=start_date_str, end_date=end_date_str)

        if df is None or df.empty:
            raise ValueError("No data returned from nsepython.")

        # --- Data Cleaning and Selection ---
        # Select relevant columns (adjust if column names differ slightly in future versions)
        if 'CH_TIMESTAMP' not in df.columns or 'CH_CLOSING_PRICE' not in df.columns:
             raise ValueError("Expected columns ('CH_TIMESTAMP', 'CH_CLOSING_PRICE') not found in the data.")

        df_processed = df[['CH_TIMESTAMP', 'CH_CLOSING_PRICE']].copy()

        # Rename columns for consistency
        df_processed.rename(columns={'CH_TIMESTAMP': 'date', 'CH_CLOSING_PRICE': 'close'}, inplace=True)

        # Convert 'date' column to datetime objects
        df_processed['date'] = pd.to_datetime(df_processed['date'])

        # Convert 'close' price to numeric, handling potential errors
        df_processed['close'] = pd.to_numeric(df_processed['close'], errors='coerce')
        df_processed.dropna(subset=['close'], inplace=True) # Remove rows where close price couldn't be converted

        # Sort values by date in ascending order (important for time series)
        df_processed = df_processed.sort_values('date').reset_index(drop=True)

        print(f"✅ Fetched and processed {len(df_processed)} records!\n")
        return df_processed

    except Exception as e:
        print(f"🚨 Error fetching or processing data from nsepython: {e}")
        return None

# Step 2: Preprocess data (No changes needed)
def preprocess_data(data, n_steps=60):
    print("🧪 Preprocessing data...")
    close_prices = data['close'].values.reshape(-1, 1)

    # Check if close_prices has enough data
    if len(close_prices) <= n_steps:
        print(f"🚨 Error: Not enough data ({len(close_prices)} points) to create sequences with n_steps={n_steps}")
        return None, None, None, None # Return None values to indicate failure

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(n_steps, len(scaled)):
        X.append(scaled[i - n_steps:i, 0])
        y.append(scaled[i, 0])

    # Check if X and y were populated
    if not X or not y:
         print(f"🚨 Error: Could not create sequences. Check data length and n_steps.")
         return None, None, None, None

    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    print(f"📊 Shape of X: {X.shape}, y: {y.shape}")
    return X, y, scaler, scaled

# Step 3: Build model (No changes needed)
def build_model(input_shape):
    print("\n🧠 Building model...")
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("✅ Model ready!\n")
    return model

# Step 4: Predict next 7 days (No changes needed)
def predict_next_days(model, scaled_data, scaler, n_steps=60, days=7):
    print(f"📈 Predicting next {days} days...")
    if len(scaled_data) < n_steps:
         print(f"🚨 Error: Not enough historical data ({len(scaled_data)}) in scaled_data to make prediction with n_steps={n_steps}")
         return None

    input_seq = scaled_data[-n_steps:]
    predictions = []

    for _ in range(days):
        if input_seq.shape != (n_steps, 1): # Add safety check
             input_seq = input_seq.reshape(n_steps, 1)

        input_reshaped = input_seq.reshape(1, n_steps, 1)
        pred = model.predict(input_reshaped, verbose=0)[0, 0]
        predictions.append(pred)
        # Update the input sequence for the next prediction
        input_seq = np.append(input_seq[1:], [[pred]], axis=0) # Correct way to append for shape (n_steps, 1)

    # Check if predictions were generated
    if not predictions:
         print("🚨 Error: No predictions were generated.")
         return None

    predictions_array = np.array(predictions).reshape(-1, 1)
    # Inverse transform predictions
    try:
        inversed_predictions = scaler.inverse_transform(predictions_array)
        return inversed_predictions
    except Exception as e:
        print(f"🚨 Error during inverse transform: {e}")
        return None


# Step 5: Plotting (No changes needed)
def plot_predictions(preds, last_date, symbol):
    """
    Generates and saves a dark-themed plot of stock predictions.

    Args:
        preds (list or np.array): List of predicted prices.
        last_date (datetime or pd.Timestamp): The last date from historical data.
        symbol (str): The stock symbol for the title and filename.

    Returns:
        str: The filename of the saved plot, or None if plotting fails.
    """
    try:
        # --- Date Handling (Ensure last_date is usable) ---
        if isinstance(last_date, datetime):
            last_date = pd.Timestamp(last_date)
        elif not isinstance(last_date, pd.Timestamp):
            print(f"⚠️ Warning: last_date type ({type(last_date)}) might not be optimal for date_range. Converting.")
            try:
                last_date = pd.to_datetime(last_date)
            except Exception as e:
                print(f"🚨 Error converting last_date: {e}. Plotting failed.")
                return None

        # --- Generate Dates for X-axis ---
        # Using Business Day frequency
        dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(preds), freq='B')

        # --- Define Dark Theme Colors ---
        dark_bg = '#1c1c1c'
        light_fg = '#e0e0e0' # For text, ticks, main axes spines
        grid_color = '#444444' # Subdued grid lines
        line_color = '#29b6f6' # A bright cyan/light blue for the plot line
        marker_color = '#fdd835' # A contrasting yellow for markers (optional)

        # --- Create Plot using Object-Oriented Interface ---
        fig, ax = plt.subplots(figsize=(12, 6)) # Create figure and axes object

        # --- Apply Dark Theme Styles ---
        fig.patch.set_facecolor(dark_bg)  # Set figure background color
        ax.set_facecolor(dark_bg)        # Set axes background color

        # Plot the prediction data
        ax.plot(dates, preds,
                marker='o',          # Add markers
                markersize=5,        # Size of markers
                markerfacecolor=marker_color, # Marker fill color
                markeredgecolor=dark_bg,    # Marker edge color (match bg)
                linestyle='-',       # Solid line
                linewidth=1.5,       # Line width
                color=line_color,    # Line color
                label='Predicted Prices')

        ax.spines['top'].set_color(grid_color)
        ax.spines['right'].set_color(grid_color)
        ax.spines['bottom'].set_color(light_fg) # Make bottom axis visible
        ax.spines['left'].set_color(light_fg)   # Make left axis visible

        ax.tick_params(axis='x', colors=light_fg, labelsize=10)
        ax.tick_params(axis='y', colors=light_fg, labelsize=10)

        ax.set_xlabel("Date", color=light_fg, fontsize=12)
        ax.set_ylabel("Price (₹)", color=light_fg, fontsize=12) # Assuming Rupee

        ax.grid(True, color=grid_color, linestyle='--', linewidth=0.5, alpha=0.7)

        legend = ax.legend(facecolor='#2a2a2a', edgecolor=grid_color, fontsize=10, framealpha=0.8)
        for text in legend.get_texts():
            text.set_color(light_fg) 

        fig.autofmt_xdate()

        output_dir = "predictions"
        os.makedirs(output_dir, exist_ok=True)

        timestamp_str = datetime.now().strftime('%Y-%m-%d')
        filename = os.path.join(output_dir, f"{symbol}_{timestamp_str}.png")

        plt.savefig(
            filename,
            dpi=300,                     
            bbox_inches='tight',         
            facecolor=fig.get_facecolor() 
            )
        plt.close(fig)

        print(f"📊 Saved dark theme plot to {filename}")
        return filename

    except Exception as e:
        print(f"🚨 An error occurred during plotting: {e}")
        try:
            plt.close(fig)
        except NameError:
            pass
        return None



def get_stock_predictions(symbol: str, days_to_predict: int = 7, n_steps: int = 60, epochs: int = 90, days_history: int = 750):

    print(f"\n--- Generating prediction for {symbol} ---")
    symbol = symbol.strip().upper() # Clean up symbol

    # 1. Fetch Data
    df = fetch_data_nsepython(symbol, days_history=days_history)

    if df is None or df.empty:
        print(f"❌ Could not fetch data for {symbol}.")
        return None # Indicate failure

    # Store last date before potentially failing preprocessing
    last_day = df['date'].max()

    # 2. Preprocess Data
    X, y, scaler, scaled_data = preprocess_data(df, n_steps=n_steps)

    if X is None: # Check if preprocessing failed
        print(f"❌ Preprocessing failed for {symbol}.")
        return None # Indicate failure

    # 3. Build Model
    # Input shape depends on the actual preprocessed data shape
    model = build_model((X.shape[1], 1))

    # 4. Train Model
    print(f"🧠 Training model for {symbol}...")
    try:
        # Consider adding validation data if possible for better training practices
        model.fit(X, y, epochs=epochs, batch_size=64, verbose=0) # Set verbose=0 for backend use
        print(f"✅ Training complete for {symbol}!")
    except Exception as e:
        print(f"🚨 Error during model training for {symbol}: {e}")
        return None

    # 5. Predict Future Days
    predictions_raw = predict_next_days(model, scaled_data, scaler, n_steps, days=days_to_predict)

    if predictions_raw is None:
        print(f"❌ Prediction failed for {symbol}.")
        return None 
    filename = plot_predictions(predictions_raw, last_day, symbol)

    predictions_list = [round(float(p[0]), 2) for p in predictions_raw]

    print(f"✅ Predictions generated for {symbol}: {predictions_list}")

    
    return {
        'predictions': predictions_list,
        'last_date': last_day,
        'filename': f"{filename}"
    }
    
# Main logic
if __name__ == "__main__":
    # Use upper() for consistency, strip whitespace
    symbol = input("Enter stock symbol (e.g., SBIN, RELIANCE, INFY): ").strip().upper()
    # Note: For NSE stocks, sometimes ".NS" is needed for other APIs, but nsepython often handles the base symbol.

    # Fetch data using the new function
    # Request more history to account for non-trading days (e.g., 750 days to get ~500 trading days)
    df = fetch_data_nsepython(symbol, days_history=750)

    if df is not None and not df.empty:
        n_steps = 60
        X, y, scaler, scaled_data = preprocess_data(df, n_steps=n_steps)

        # Proceed only if preprocessing was successful
        if X is not None and y is not None and scaler is not None and scaled_data is not None:
            model = build_model((X.shape[1], 1)) # input shape based on preprocessed X

            print("🧠 Training model...")
            # Consider adding validation_split or a separate validation set for better training
            model.fit(X, y, epochs=90, batch_size=64, verbose=1)
            print("✅ Training complete!\n")

            predictions = predict_next_days(model, scaled_data, scaler, n_steps, days=7)

            if predictions is not None:
                print("📤 Predicted prices for next 7 business days:")
                for i, val in enumerate(predictions, 1):
                    # Ensure val is indexable, handle potential shape issues
                    price = val[0] if isinstance(val, (list, np.ndarray)) and len(val) > 0 else val
                    print(f"Day {i}: ₹{price:.2f}") # Using Rupee symbol for NSE

                # Get the last date from the fetched data for plotting
                last_day = df['date'].max()
                plot_predictions(predictions, last_day)
            else:
                print("❌ Prediction failed.")
        else:
            print("❌ Preprocessing failed. Cannot train or predict.")
    else:
        print(f"❌ Could not fetch data for {symbol}. Exiting.")