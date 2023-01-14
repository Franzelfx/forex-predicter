import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
from keras.models import Sequential
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed, Conv1D, MaxPooling1D, Flatten

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
PAIR = 'GBPUSD'
TOKEN = 'kvtkOoyqcuTgNrBqRGIhhLe766CLYbpo'
MINUTES = 15
pairs = ['EURUSD', 'GBPUSD', 'AUDUSD', 'GBPJPY', 'EURJPY', 'EURGBP', 'USDJPY', 'USDCHF', 'USDCAD']

# Split sequence should split inout into an vector with
# volume and close and the output in a vector with close only
def split_sequence(sequence, n_steps, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # Find the end of this pattern
        end_ix = i + n_steps
        out_end_ix = end_ix + n_steps_out
        # Check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # Gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix:out_end_ix, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y).reshape(-1, n_steps_out, 1)

def main():
    # Loop over all pairs
    for pair in pairs:
        # Get the data
        print(f"Processing pair: {pair}")
        proceed(pair)

def model_1(n_steps_in, n_steps_out, n_features, units=64):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=16, activation='tanh', input_shape=(n_steps_in, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(round(units / 2), activation='tanh', return_sequences=True))
    model.add(LSTM(units, activation='tanh', return_sequences=True))
    model.add(TimeDistributed(Dense(units, activation='tanh')))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(units, activation='tanh')))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(units, activation='tanh')))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(1, activation='linear')))
    model.build(input_shape=(n_steps_in, n_features))
    return model

def plot_candles(df, pair_name):
    # Plot the data as candles plus the volume
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'])])
    fig.write_image(f"../pairs/chart/{pair_name}_chart.png", width=1920, height=1080, scale=1)

def plot_predictions(n_steps_in, n_steps_out, train, test, y_hat, pair_name):
    # Plot the last n_steps in of the close price and the prediction/test (x of prediction starts at the last n_steps_in)
    plt.clf()
    # Plot the train data
    plt.plot(np.arange(0, n_steps_in), train[-n_steps_in:], label='Close')
    # Plot test and prediction
    plt.plot(np.arange(n_steps_in, n_steps_in + n_steps_out), y_hat, label='Prediction')
    # Plot test
    plt.plot(np.arange(n_steps_in, n_steps_in + n_steps_out), test[:n_steps_out], label='Test')
    plt.legend()
    plt.savefig(f'../predictions/{pair_name}_test.png')
    plt.cla()

def plot_loss(loss, val_loss, pair_name):
    plt.figure(figsize=(10, 10))
    plt.plot(loss, label='loss')
    plt.plot(val_loss, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [Close]')
    plt.legend()
    plt.savefig(f'../train/{pair_name}_train.png', format='png')
    plt.cla()

def get_data(pair):
    # Get date in form of YYYY-MM-DD
    # 1221ef5bfc04446c9c581203bce63db0
    date_end = datetime.now().strftime("%Y-%m-%d")
    date_start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    url_1 = f'https://api.polygon.io/v2/aggs/ticker/C:{pair}/range/{MINUTES}/minute/{date_start}/{date_end}?adjusted=true&sort=asc&limit=50000&apiKey={TOKEN}'
    data_1 = requests.get(url_1).json()

    date_start = (datetime.now() - timedelta(days=50)).strftime("%Y-%m-%d")
    date_end = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    url_2 = f'https://api.polygon.io/v2/aggs/ticker/C:{pair}/range/{MINUTES}/minute/{date_start}/{date_end}?adjusted=true&sort=asc&limit=50000&apiKey={TOKEN}'
    data_2 = requests.get(url_2).json()

    date_start = (datetime.now() - timedelta(days=80)).strftime("%Y-%m-%d")
    date_end = (datetime.now() - timedelta(days=50)).strftime("%Y-%m-%d")
    url_3 = f'https://api.polygon.io/v2/aggs/ticker/C:{pair}/range/{MINUTES}/minute/{date_start}/{date_end}?adjusted=true&sort=asc&limit=50000&apiKey={TOKEN}'
    data_3 = requests.get(url_3).json()
    if not 'results' in (data_1 and data_2 and data_3):
        print('Data from csv')
        df = pd.read_csv(f'../pairs/{pair}.csv')
    else:
        print('Data from api')
        df_1 = pd.DataFrame(data_1['results'])
        df_2 = pd.DataFrame(data_2['results'])
        df_3 = pd.DataFrame(data_3['results'])
        df = pd.concat([df_1, df_2, df_3], ignore_index=True)
        df = df.sort_values(by=['t'])
        # Convert to dataframe
        df['t'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('t', inplace=True)
        df = df[['o', 'h', 'l', 'c', 'v', 'vw']]
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Volume_Weighted']
        # Add a row with the indexes from 0 to len(df)
        df['Index'] = np.arange(0, len(df))
        # Safe to csv
        df.to_csv(f'../pairs/{pair}.csv')
    return df

def on_balance_volume(close, volume):
    obv = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    return obv

def accumulate_distribution(close, low, high):
    ad = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            ad[i] = ad[i-1] + (close[i] - low[i]) - (high[i] - close[i])
        elif close[i] < close[i-1]:
            ad[i] = ad[i-1] - (close[i] - low[i]) + (high[i] - close[i])
        else:
            ad[i] = ad[i-1]
    return ad

def relative_strength_index(close, n):
    rsi = np.zeros(len(close))
    for i in range(n, len(close)):
        up = 0
        down = 0
        for j in range(i-n, i):
            if close[j] > close[j-1]:
                up += close[j] - close[j-1]
            elif close[j] < close[j-1]:
                down += close[j-1] - close[j]
        if down == 0:
            rsi[i] = 100
        else:
            rsi[i] = 100 - (100 / (1 + (up / down)))
    return rsi

def stochastic_oscillator(close, low, high, n):
    stoch = np.zeros(len(close))
    for i in range(n, len(close)):
        stoch[i] = 100 * ((close[i] - min(low[i-n:i])) / (max(high[i-n:i]) - min(low[i-n:i])))
    return stoch

def get_model_dataset(df, n_steps_out):
    # extract close price and volume as input features
    indicator_window = 50
    moving_average = df['Close'].rolling(window=indicator_window).mean().values
    _open = df['Open'].values
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    volume = df['Volume'].values
    volume_weighted = df['Volume_Weighted'].values
    # Calculate on balance volume indicator
    obv = on_balance_volume(close, volume)
    # Calculate Accumulation/Distribution indicator
    ad = accumulate_distribution(close, low, high)
    # Calculate RSI indicator
    rsi = relative_strength_index(close, indicator_window)
    # Calculate Stochastic Oscillator indicator
    stoch = stochastic_oscillator(close, low, high, indicator_window)

    # Remove first 50 rows
    _open = _open[indicator_window:]
    high = high[indicator_window:]
    low = low[indicator_window:]
    close = close[indicator_window:]
    volume = volume[indicator_window:]
    volume_weighted = volume_weighted[indicator_window:]
    moving_average = moving_average[indicator_window:]
    obv = obv[indicator_window:]
    ad = ad[indicator_window:]
    rsi = rsi[indicator_window:]
    stoch = stoch[indicator_window:]

    # Split into train and test
    test_size = n_steps_out
    train_size = len(close) - test_size
    _open, test_open = _open[0:train_size], _open[train_size:len(_open)]
    high, test_high = high[0:train_size], high[train_size:len(high)]
    low, test_low = low[0:train_size], low[train_size:len(low)]
    close, test_close = close[0:train_size], close[train_size:len(close)]
    volume, test_volume = volume[0:train_size], volume[train_size:len(volume)]
    volume_weighted, test_volume_weighted = volume_weighted[0:train_size], volume_weighted[train_size:len(volume_weighted)]
    moving_average, test_moving_average = moving_average[0:train_size], moving_average[train_size:len(moving_average)]
    obv, test_obv = obv[0:train_size], obv[train_size:len(obv)]
    ad, test_ad = ad[0:train_size], ad[train_size:len(ad)]
    rsi, test_rsi = rsi[0:train_size], rsi[train_size:len(rsi)]
    stoch, test_stoch = stoch[0:train_size], stoch[train_size:len(stoch)]

    # Scale volume and close price
    scaler_open = MinMaxScaler(feature_range=(-1, 1))
    scaler_high = MinMaxScaler(feature_range=(-1, 1))
    scaler_low = MinMaxScaler(feature_range=(-1, 1))
    scaler_close = MinMaxScaler(feature_range=(-1, 1))
    scaler_volume = MinMaxScaler(feature_range=(-1, 1))
    scaler_volume_weighted = MinMaxScaler(feature_range=(-1, 1))
    scaler_moving_average = MinMaxScaler(feature_range=(-1, 1))
    scaler_obv = MinMaxScaler(feature_range=(-1, 1))
    scaler_ad = MinMaxScaler(feature_range=(-1, 1))
    scaler_rsi = MinMaxScaler(feature_range=(-1, 1))
    scaler_stoch = MinMaxScaler(feature_range=(-1, 1))

    # Fit the scaler
    _open = scaler_open.fit_transform(_open.reshape(-1, 1))
    high = scaler_high.fit_transform(high.reshape(-1, 1))
    low = scaler_low.fit_transform(low.reshape(-1, 1))
    close = scaler_close.fit_transform(close.reshape(-1, 1))
    volume = scaler_volume.fit_transform(volume.reshape(-1, 1))
    volume_weighted = scaler_volume_weighted.fit_transform(volume_weighted.reshape(-1, 1))
    moving_average = scaler_moving_average.fit_transform(moving_average.reshape(-1, 1))
    obv = scaler_obv.fit_transform(obv.reshape(-1, 1))
    ad = scaler_ad.fit_transform(ad.reshape(-1, 1))
    rsi = scaler_rsi.fit_transform(rsi.reshape(-1, 1))
    stoch = scaler_stoch.fit_transform(stoch.reshape(-1, 1))

    # Convert to numpy arrays
    in_seq1 = np.array(_open)
    in_seq2 = np.array(high)
    in_seq3 = np.array(low)
    in_seq4 = np.array(close)
    in_seq5 = np.array(volume)
    in_seq6 = np.array(volume_weighted)
    in_seq7 = np.array(moving_average)
    in_seq8 = np.array(obv)
    in_seq9 = np.array(ad)
    in_seq10 = np.array(rsi)
    in_seq11 = np.array(stoch)

    # Convert to 2D array
    in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    in_seq2 = in_seq2.reshape((len(in_seq2), 1))
    in_seq3 = in_seq3.reshape((len(in_seq3), 1))
    in_seq4 = in_seq4.reshape((len(in_seq4), 1))
    in_seq5 = in_seq5.reshape((len(in_seq5), 1))
    in_seq6 = in_seq6.reshape((len(in_seq6), 1))
    in_seq7 = in_seq7.reshape((len(in_seq7), 1))
    in_seq8 = in_seq8.reshape((len(in_seq8), 1))
    in_seq9 = in_seq9.reshape((len(in_seq9), 1))
    in_seq10 = in_seq10.reshape((len(in_seq10), 1))
    in_seq11 = in_seq11.reshape((len(in_seq11), 1))
    # Horizontal stack inputs
    dataset = np.hstack((in_seq1, in_seq5, in_seq7))
    # Print shapes
    print(dataset.shape)
    return dataset, _open, test_open, scaler_open

def proceed(pair: str):
    n_steps_in = 60
    n_steps_out = 30
    # Get the data from the API or from CSV
    df = get_data(pair)
    # Plot the data as candles plus the volume
    plot_candles(df, pair)
    # Check for nan values
    dataset, _open, test_open, scaler_open = get_model_dataset(df, n_steps_out)
    print(np.isnan(dataset).any())
    # Convert into input/output
    X, y = split_sequence(dataset, n_steps_in, n_steps_out)
    # Print shapes
    print(X.shape, y.shape)
    # The dataset knows the number of features, e.g. 2
    n_features = X.shape[2]
    # Define model
    model = model_1(n_steps_in, n_steps_out, n_features, units=64)
    #Fit model
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='mae')
    model.summary()
    fit = model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)
    # Plot loss
    plot_loss(fit.history['loss'], fit.history['val_loss'], pair)
    # Take n_steps_in from the last n_steps_in of the dataset
    x_input = dataset[-n_steps_in:, :]
    # Reshape to [1, n_steps_in, n_features]
    x_input = x_input.reshape((1, n_steps_in, n_features))
    # Predict
    yhat = model.predict(x_input)
    # Invers transform
    y_hat = scaler_open.inverse_transform(yhat.reshape(-1, 1))
    _open = scaler_open.inverse_transform(_open.reshape(-1, 1))
    # Plot the last n_steps in of the close price and the prediction/test (x of prediction starts at the last n_steps_in)
    plot_predictions(n_steps_in, n_steps_out, _open, test_open, y_hat, pair)


if __name__ == '__main__':
    main()