from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
from keras.models import Sequential
from keras.models import Model
import keras.layers
from keras.layers import Dense, Input, LSTM, Concatenate, LeakyReLU
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from math import sqrt
from matplotlib import pyplot
from numpy import array, concatenate
import numpy as np
import os


# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, n_features=1,
                         pred_id=0, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    # Drop non-predicting variables
    drop_cols = list(set(range(-n_features * n_out, 0)) - \
                     set([(-(i + 1) * n_features) for i in range(n_out)]))

    agg = agg.drop(agg.columns[drop_cols], axis=1)
    return agg

# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq, n_features, pred_id=0):
    # extract raw values
    raw_values = series.values
    raw_values = raw_values.reshape(len(raw_values), -1)
    print(raw_values.shape)

    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(raw_values)
    scaled_values = scaled_values.reshape(len(scaled_values), -1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq,
                                      n_features=n_features,
                                      pred_id=pred_id)
    supervised_values = supervised.values
    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test


# transform series into train and test sets for supervised learning
def prepare_all_state_data(states, series, n_test, n_lag, n_seq,
                           nth_week, n_features, n_out=7, pred_id=0):
    train = np.empty((0, n_lag * n_features + n_seq))
    test = np.empty((0, n_lag * n_features + n_seq))

    for state in states:
        state_df = series[series['location_name'] == state]
        state_df = state_df.drop(['location', 'location_name',
                                  'hospitalizations', 'cases', 'spc'], axis=1)
        # 'spc', 'adj_cov_gst', 'sm_sym_gst'

        state_supervised = series_to_supervised(state_df, n_in=n_lag, n_out=n_seq,
                                                n_features=n_features, pred_id=0)
        state_supervised_values = state_supervised.values
        state_train, state_test = state_supervised_values[0:-n_test], state_supervised_values[-n_test:]

        train_unclean = np.append(train, state_train, axis=0)
        test = np.append(test, state_test, axis=0)
        # all_state_supervised = all_state_supervised.append(state_supervised)
        # print(f'{state} is statcked in DataFrame.')

        # print(train.shape, test.shape)
        # # Clean up all data for validation
        train = train_unclean[:-(n_out * nth_week) + 1, :]  # if -(n_out*nth_week)+2, leaking first day
    print('Deleted rows: ', n_out * nth_week)
    print('Uncleaned Train shape: ', (len(states) * (state_supervised.shape[0] - 1),
                                      state_supervised.shape[1]))
    print('Cleaned Train shape: ', train.shape)

    raw_values = np.concatenate((train, test), axis=0)
    print(f'Concatenated Dataframe shape: {raw_values.shape}')
    print(raw_values[1, -(nth_week + 1) * n_out:])
    if nth_week >= 2:
        raw_values = np.delete(raw_values, np.s_[n_features * n_lag: n_features * n_lag + n_out * (nth_week - 1)],
                               axis=1)
    print(f'Concatenated Dataframe shape of {nth_week}th week: {raw_values.shape}')
    print(raw_values[1, -2 * n_out:])
    # # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(raw_values)
    scaled_values = scaled_values.reshape(len(scaled_values), -1)

    # split into train and test sets
    train, test = scaled_values[0:-len(test)], scaled_values[-len(test):]
    return scaler, train, test


class MultiQuantileLoss(tf.keras.losses.Loss):

    def __init__(self, quantiles: list, **kwargs):
        super(MultiQuantileLoss, self).__init__(**kwargs)

        self.quantiles = quantiles

    def call(self, y_true, y_pred):
        # get quantile value
        q_id = int(y_pred.name.split("/")[1][1:])
        q = self.quantiles[q_id]

        # minimize quantile error
        q_error = tf.subtract(y_true, y_pred)
        q_loss = tf.reduce_mean(tf.maximum(q * q_error, (q - 1) * q_error), axis=-1)
        return q_loss


# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_features,
             n_seq, n_batch, nb_epoch,
             n_neurons):
    # quantiles = [.1, .5, .9]
    quantiles = [.025, .100, .250, .500, .750, .900, .975]
    # quantiles = [.010, .025, .050, .100, .150, .200, .250, .300, .350, .400, .450, .500,
    #              .550, .600, .650, .700, .750, .800, .850, .900, .950, .975, .990]

    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag * n_features], train[:, n_lag * n_features:]
    X = X.reshape(X.shape[0], n_lag, n_features)
    # design network
    model = Sequential()
    model.add(LSTM(n_neurons, activation='tanh',
                   return_sequences=True,
                   input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(n_neurons, activation='tanh'))
    model.add(Dense(y.shape[1]))
    # loss=tf.keras.losses.Huber(),mae, mse
    model.compile(loss=tf.keras.losses.Huber(), optimizer='adam')
    # fit network
    history = model.fit(X, y, epochs=nb_epoch,
                        batch_size=n_batch,
                        verbose=2, shuffle=False)

    return model


# fit an LSTM network to training data
def fit_lstm_quantile(train1, train2, n_lag1, n_lag2,
                      n_features, n_out, n_batch, nb_epoch,
                      n_neurons, val=False, val_1=None, val_2=None):
    # quantiles = [.025, .100, .250, .500, .750, .900, .975]
    # quantiles = [.050, .500, .950]
    quantiles = [.010, .025, .050, .100, .150, .200, .250, .300, .350, .400, .450, .500,
                 .550, .600, .650, .700, .750, .800, .850, .900, .950, .975, .990]
    output_dim = len(quantiles)
    print(output_dim)
    outputs = []

    # Define a learnable weights
    W = tf.Variable(1.0, trainable=True)

    # reshape training into [samples, timesteps, features]
    X1, y1 = train1[:, 0:n_lag1 * n_features], train1[:, n_lag1 * n_features:]
    X1 = X1.reshape(X1.shape[0], n_lag1, n_features)
    # reshape training into [samples, timesteps, features]
    X2, y2 = train2[:, 0:n_lag2 * n_features], train2[:, n_lag2 * n_features:]
    X2 = X2.reshape(X2.shape[0], n_lag2, n_features)

    # design network
    ts_input1 = Input((X1.shape[1], X1.shape[2]))
    lstm1_1_out = LSTM(int(n_neurons), activation="tanh", return_sequences=True)(ts_input1)
    lstm1_2_out = LSTM(int(n_neurons / 2), activation="tanh", return_sequences=True)(lstm1_1_out)
    lstm1_3_out = LSTM(int(n_neurons / 2), activation='tanh', return_sequences=True)(lstm1_2_out)
    lstm1_4_out = LSTM(int(n_neurons / 2), activation='tanh')(lstm1_3_out)
    # lstm1_4_out = LSTM(int(n_neurons/2), activation='tanh')(lstm1_3_out)
    map_lstm1 = Dense(64)(lstm1_4_out)

    # design network
    ts_input2 = Input((X2.shape[1], X2.shape[2]))
    lstm2_1_out = LSTM(int(n_neurons), activation="tanh", return_sequences=True)(ts_input2)
    lstm2_2_out = LSTM(int(n_neurons / 2), activation="tanh", return_sequences=True)(lstm2_1_out)
    lstm2_3_out = LSTM(int(n_neurons / 2), activation='tanh', return_sequences=True)(lstm2_2_out)
    lstm2_4_out = LSTM(int(n_neurons / 2), activation='tanh')(lstm2_3_out)
    # lstm2_4_out = LSTM(int(n_neurons/2), activation='tanh')(lstm2_3_out)
    map_lstm2 = Dense(64)(lstm2_4_out)

    # Concatenate LSTMs
    # lstm_2_out = keras.layers.concatenate([w * map_lstm1, map_lstm2])
    lstm_2_out = keras.layers.Concatenate()([W * map_lstm1, map_lstm2])
    print(f'W: {W}')
    # dense_1_out = Dense(96)(lstm_2_out)

    #     # create an output for each quantile
    #     for i,quantile in enumerate(quantiles):
    #         output = Dense(y.shape[1], name="q_"+str(i))(dense_1_out)
    #         outputs.append(output)
    outputs = [Dense(n_out, activation='linear', name="q%d" % q_i)(lstm_2_out) for q_i in range(output_dim)]

    # loss
    q_loss = MultiQuantileLoss(quantiles)

    # intialize & compile
    model = Model(inputs=[ts_input1, ts_input2], outputs=outputs)

    # Set Learning rate decay for AdamOpt
    initial_lr = 0.0008  # Default for adam is 0.001
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_lr,
    #     decay_steps=3000,
    #     decay_rate=0.96,
    #     staircase=True
    #     )
    Adam_Opt = tf.keras.optimizers.Adam(initial_lr)

    # Compile the model
    model.compile(optimizer=Adam_Opt, loss=q_loss.call)
    # Huber Loss
    # model.compile(optimizer=Adam_Opt, loss=tf.keras.losses.Huber())

    print(y1.shape)
    logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10,
                       restore_best_weights=False)

    if val != False:
        print('Has Validation!')
        # Random Validation
        if val == 'Random':
            print('Random Validation Split!')
            val_pct = 0.05
            # Validation Preparation
            ## Preparing for validation set
            val_idx = np.random.randint(0, len(train1), int(val_pct * len(train1)))

            # val_1 = train1[val_idx]
            # val_2 = train2[val_idx]

            # train1 = np.delete(train1, val_idx, axis=0)
            # train2 = np.delete(train2, val_idx, axis=0)

            # val_1, val_2 = train1[-1,], train2[-1,]
            val_X1, val_y = val_1[:, 0:n_lag1 * n_features], val_1[:, n_lag1 * n_features:]
            val_X1 = val_X1.reshape((val_X1.shape[0], n_lag1, n_features))

            val_X2, val_y = val_2[:, 0:n_lag2 * n_features], val_2[:, n_lag2 * n_features:]
            val_X2 = val_X2.reshape((val_X2.shape[0], n_lag2, n_features))

            # fit network
            history = model.fit([X1, X2], y1, epochs=nb_epoch,
                                batch_size=n_batch,
                                verbose=2, shuffle=False,
                                # validation_split=0.01,
                                validation_data=([val_X1, val_X2], val_y),
                                callbacks=[tensorboard_callback, es])
        elif val == 'Recent':
            print('Recent Validation Split!')
            train_idx = []
            for i in range(51):
                for j in range(51):
                    train_idx = train_idx + [int(i + (j * train1.shape[0] / 51))]

            train1 = train1[train_idx]
            train2 = train2[train_idx]

            # fit network
            history = model.fit([X1, X2], y1, epochs=nb_epoch,
                                batch_size=n_batch,
                                verbose=2, shuffle=False,
                                validation_split=0.05,
                                # validation_data=([val_X1, val_X2], val_y),
                                callbacks=[tensorboard_callback, es])
        else:
            print('Default (State) Validation Split!')
            print(f'Validation Pct. : {val}')
            # Default Validation Split
            # Last state's data as validation
            # Validation Preparation
            ## Preparing for validation set
            # val_idx = [int((-1) - n * (train1.shape[0] / 51 )) for n in range(51)]

            # train1 = np.delete(train1, val_idx, axis=0)
            # train2 = np.delete(train2, val_idx, axis=0)

            # val_1 = train1[val_idx]
            # val_2 = train2[val_idx]
            # # val_1, val_2 = train1[-1,], train2[-1,]
            # val_X1, val_y = val_1[:, 0:n_lag1*n_features], val_1[:, n_lag1*n_features:]
            # val_X1 = val_X1.reshape((val_X1.shape[0], n_lag1, n_features))

            # val_X2, val_y = val_2[:, 0:n_lag2*n_features], val_2[:, n_lag2*n_features:]
            # val_X2 = val_X2.reshape((val_X2.shape[0], n_lag2, n_features))

            # fit network
            history = model.fit([X1, X2], y1, epochs=nb_epoch,
                                batch_size=n_batch,
                                verbose=2, shuffle=False,
                                validation_split=val,
                                # validation_data=([val_X1, val_X2], val_y),
                                callbacks=[tensorboard_callback, es])
    else:
        print('No Validation!')
        history = model.fit([X1, X2], y1, epochs=nb_epoch,
                            batch_size=n_batch,
                            verbose=2, shuffle=False,
                            callbacks=[tensorboard_callback, es])

    return model


# evaluate the persistence model
def make_forecasts(model, n_batch, test1, test2,
                   n_features, n_lag1, n_lag2, n_seq):
    test_X1, test_y = test1[:, 0:n_lag1 * n_features], test1[:, n_lag1 * n_features:]
    test_X1 = test_X1.reshape((test_X1.shape[0], n_lag1, n_features))

    test_X2, test_y = test2[:, 0:n_lag2 * n_features], test2[:, n_lag2 * n_features:]
    test_X2 = test_X2.reshape((test_X2.shape[0], n_lag2, n_features))

    forecasts = model.predict([test_X1, test_X2], batch_size=n_batch)
    # print(forecasts.shape)

    return forecasts


# inverse data transform on forecasts
def inverse_transform(test, forecasts, scaler,
                      n_lag, n_features):
    test_X, test_y = test[:, 0:n_lag * n_features], test[:, n_lag * n_features:]
    inverted_hat = np.zeros(shape=forecasts.shape)
    inverted_truth = np.zeros(shape=test_y.shape)
    for i in range(forecasts.shape[1]):
        # Inverse predictions
        y_hat_i = forecasts[:, [i]]
        inv_yhat_i = concatenate((y_hat_i, test_X[:, -(n_features - 1):]), axis=1)
        inv_yhat_i = scaler.inverse_transform(inv_yhat_i)
        inv_yhat_i = inv_yhat_i[:, [0]]

        # store
        inverted_hat[:, [i]] = inv_yhat_i

        # Inverse Ground Truth
        y_i = test_y[:, [i]]
        inv_y_i = concatenate((y_i, test_X[:, -(n_features - 1):]), axis=1)
        inv_y_i = scaler.inverse_transform(inv_y_i)
        inv_y_i = inv_y_i[:, [0]]

        # store
        inverted_truth[:, [i]] = inv_y_i
    return inverted_hat, inverted_truth


# inverse data transform on forecasts
def inverse_transform_all_state(test, forecasts, scaler,
                                n_lag, n_features, n_seq, quantile_i):
    test_X, test_y = test[:, 0:n_lag * n_features], test[:, n_lag * n_features:]
    # print(test_X.shape, test_y.shape)
    forecasts_median = forecasts[quantile_i]
    inverted_hat = concatenate((test_X, forecasts_median), axis=1)
    inverted_hat = scaler.inverse_transform(inverted_hat)
    inverted_hat = inverted_hat[:, -n_seq:]

    inverted_truth = scaler.inverse_transform(test)
    inverted_truth = inverted_truth[:, -n_seq:]

    return inverted_hat, inverted_truth


def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    mae_list = []
    for i in range(n_seq):
        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]

        np_actual = np.array(actual)
        np_predicted = np.array(predicted)
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs((np_actual - np_predicted) / actual)) * 100
        rmse = sqrt(mean_squared_error(actual, predicted))

        mae_list.append(mae)

        print('t+%d MAE: %f' % ((i + 1), mae))
        print('t+%d MAPE: %f' % ((i + 1), mape))
        print('t+%d RMSE: %f' % ((i + 1), rmse))
        print('==============================')

    return mae_list

# Export forecasts according to Hub's format
# Input: inv_forecast = [np.array(num_state, num_day_forecast) * 23_quantile]
def export_forecasts(inv_forecast, date, n_seq, states_list, train_date):
    # output name format = YYYY-MM-DD-team-model.csv
    output_name = date.strftime('%Y-%m-%d') + '-' + 'CUBoulder' + '-' + 'SLSTM_214_11.csv'

    # Define fips to state
    fip_state = {
        "01": "Alabama",
        "02": "Alaska",
        "04": "Arizona",
        "05": "Arkansas",
        "06": "California",
        "08": "Colorado",
        "09": "Connecticut",
        "10": "Delaware",
        "11": "District of Columbia",
        "12": "Florida",
        "13": "Georgia",
        "15": "Hawaii",
        "16": "Idaho",
        "17": "Illinois",
        "18": "Indiana",
        "19": "Iowa",
        "20": "Kansas",
        "21": "Kentucky",
        "22": "Louisiana",
        "23": "Maine",
        "24": "Maryland",
        "25": "Massachusetts",
        "26": "Michigan",
        "27": "Minnesota",
        "28": "Mississippi",
        "29": "Missouri",
        "30": "Montana",
        "31": "Nebraska",
        "32": "Nevada",
        "33": "New Hampshire",
        "34": "New Jersey",
        "35": "New Mexico",
        "36": "New York",
        "37": "North Carolina",
        "38": "North Dakota",
        "39": "Ohio",
        "40": "Oklahoma",
        "41": "Oregon",
        "42": "Pennsylvania",
        "44": "Rhode Island",
        "45": "South Carolina",
        "46": "South Dakota",
        "47": "Tennessee",
        "48": "Texas",
        "49": "Utah",
        "50": "Vermont",
        "51": "Virginia",
        "53": "Washington",
        "54": "West Virginia",
        "55": "Wisconsin",
        "56": "Wyoming"
    }
    quantiles = [.010, .025, .050, .100, .150, .200, .250, .300, .350, .400, .450, .500,
                 .550, .600, .650, .700, .750, .800, .850, .900, .950, .975, .990]

    state_fip = {v: k for k, v in fip_state.items()}
    exported_col = ['forecast_date', 'target', 'target_end_date',
                    'location', 'type', 'quantile', 'value']
    export_df = DataFrame(columns=exported_col)
    for q in range(len(forecasts)):
        quantile = quantiles[q]
        print(f'Now formatting quantile: {quantile:.3f} || {q+1} / {len(quantiles)}')
        forecasts_quantile = inv_forecast[q]
        # i = idx of state
        for i in range(forecasts_quantile.shape[0]):
            # j = idx of days
            # print(forecasts_quantile.shape)
            for j in range(forecasts_quantile.shape[1]):
                target = f'{j+1} day ahead inc hosp'
                target_end_date = (date + timedelta(days=j+1)).strftime('%Y-%m-%d')
                location_fip = state_fip[states_list[i]]

                type = 'quantile'
                quantile_exp = quantile
                value = forecasts_quantile[i][j]
                if quantile == 0.500:
                    type_median = 'point'
                    quantile_exp_median = 'NA'

                    row_exp = [date.strftime('%Y-%m-%d'), target, target_end_date,
                               location_fip, type_median, quantile_exp_median, value]
                    export_df.loc[len(export_df.index)] = row_exp

                row_exp = [date.strftime('%Y-%m-%d'), target, target_end_date,
                           location_fip, type, quantile_exp, value]
                export_df.loc[len(export_df.index)] = row_exp

    export_df.to_csv(f'/content/gdrive/MyDrive/Omicron_evaluation_15month(non-sph)/{str(train_date.date())}/'+output_name,
                     index=False)

    print(f'{output_name} is successfully exported!')


if __name__ == '__main__':
    print('===== Start Reading Processed Data ======')
    ## Change your processed data accordingly
    df = read_csv('../data/pprocessed_df_2022-08-06.csv', header=0, parse_dates=['date'], index_col=0)
    states = list(df['location_name'].unique())
    state_pop = read_csv('./data/state_pop.csv', index_col=0)
    state_pop = state_pop.to_dict('index')

    rate_df = DataFrame()

    ## Use Mixed(High Medium Low) state as validation set
    states.append(states.pop(states.index('Kentucky')))
    states.append(states.pop(states.index('New Mexico')))
    states.append(states.pop(states.index('Vermont')))

    ## Shuffle the order of the states
    # np.random.shuffle(states)

    for state in states:
        state_df = df[df['location_name'] == state]
        state_df['smoothed_hosp'] = state_df['smoothed_hosp'] / state_pop[state]['population'] * 10000
        state_df['smoothed_cases'] = state_df['smoothed_cases'] / state_pop[state]['population'] * 10000

        rate_df = rate_df.append(state_df)
        print(f'{state} has been converted to rate.')

    mae_df = DataFrame()

    forecasts_ql_4wk = []
    inv_truth_v3_4wk = []
    mae_list_v3_4wk = []
    train_date = datetime(2022, 2, 7)

    # n loops are the num of models you want for ensemble
    for n in range(1):
        # 4 weekly models: 4*7=28
        # 2 bi-weekly models: 2*14=28
        # 1 4-weeks model: 1*28=28
        for i in range(4):
            print(f'======================== {i + 1} / 4 ========================')
            print(f'========= Current Date: {train_date + timedelta(days=(i + 1) * 7)} =========')
            wf_df = rate_df[rate_df.index >= (train_date - timedelta(days=30 * 15))]
            wf_df = wf_df[wf_df.index <= (train_date + timedelta(days=(i + 1) * 7))]
            # Configure
            n_features = 3
            n_lag = 7 * 4
            n_lag2 = 7

            n_out = 7
            n_test = 1  # len(state_supervised) - 360
            nth_week = i + 1  #
            n_seq = nth_week * 7

            # Prepare data
            scaler1, train1, test1 = prepare_all_state_data(states, wf_df, n_test=n_test,
                                                            n_lag=n_lag, n_seq=n_seq, nth_week=nth_week,
                                                            n_features=n_features, n_out=n_out, pred_id=0)

            ## If Shuffle the training data?
            # Not working, very unstable
            # np.random.shuffle(train1)

            train2, test2 = train1[:, -(n_lag2 * n_features + 7):], test1[:, -(n_lag2 * n_features + 7):]

            # print('########## Check leakage ###########')
            # for i in range(1, 14):
            #   print(np.isin(rate_df[rate_df.index==train_date+timedelta(days=i)]['smoothed_hosp'].to_list(), train1))

            # Save the scaler (Change Path if needed)
            joblib.dump(scaler1,
                        f'/content/gdrive/MyDrive/Omicron_evaluation_15month(non-sph)/{str(train_date.date())}/scaler.pkl')

            # n_feature * n_lag + n_seq
            # 4         * 21    + 7     = 91
            print("=================================================")
            print('Train1: %s, Test: %s' % (train1.shape, test1.shape))
            print('Train2: %s, Test: %s' % (train2.shape, test2.shape))

            #     X, y = train1[:, 0:n_lag*n_features], train1[:, n_lag*n_features:]
            #     X = X.reshape(X.shape[0], 1, X.shape[1])

            ################ Quantile Loss ################
            model = fit_lstm_quantile(train1, train2, n_lag1=n_lag,
                                      n_lag2=n_lag2, n_features=n_features,
                                      n_out=n_out, n_batch=2 ** 6,
                                      nb_epoch=100, n_neurons=2 ** 8,
                                      val=0.05)

            ################ Save Model ################
            model.save(
                f'/content/gdrive/MyDrive/Omicron_evaluation_15month(non-sph)/{str(train_date.date())}/week{i + 1}_model47_04')

            #     ################ Huber loss ################
            #     # model = fit_lstm(train, n_lag=n_lag, n_features=n_features,
            #     #              n_seq=n_out, n_batch=2**6, nb_epoch=100, n_neurons=96)

            print('Model training completed!')
            print('Now making forecast!')
            forecasts = make_forecasts(model, n_batch=2 ** 6,
                                       test1=test1, test2=test2,
                                       n_features=n_features,
                                       n_lag1=n_lag, n_lag2=n_lag2, n_seq=n_out)

            print('Inverse forecast to unscaled numbers!')
            # inverse transform forecasts and test
            forecasts_ql = []

            for j in range(len(forecasts)):
                print('quantile_n:', j)
                if j == len(forecasts) // 2:
                    inv_forecasts_v3, inv_truth_v3 = inverse_transform_all_state(test=test1, forecasts=forecasts,
                                                                                 scaler=scaler1, n_lag=n_lag,
                                                                                 n_features=n_features, n_seq=n_out,
                                                                                 quantile_i=j)

                    for i in range(len(inv_forecasts_v3)):
                        inv_forecasts_v3[i,] = inv_forecasts_v3[i,] / 10000 * state_pop[states[i]]['population']
                        inv_truth_v3[i,] = inv_truth_v3[i,] / 10000 * state_pop[states[i]]['population']
                        # print(i+1,'/', len(inv_forecasts_v3))
                    # print(inv_forecasts_v3.shape, inv_truth_v3.shape)

                    forecasts_ql.append(inv_forecasts_v3)

                    mae_list_v3 = evaluate_forecasts(test=inv_truth_v3, forecasts=inv_forecasts_v3,
                                                     n_lag=n_lag, n_seq=n_out)
                else:
                    print('Not Median')
                    inv_forecasts_v3, inv_truth_v3 = inverse_transform_all_state(test=test1, forecasts=forecasts,
                                                                                 scaler=scaler1, n_lag=n_lag,
                                                                                 n_features=n_features, n_seq=n_out,
                                                                                 quantile_i=j)
                    for i in range(len(inv_forecasts_v3)):
                        inv_forecasts_v3[i,] = inv_forecasts_v3[i,] / 10000 * state_pop[states[i]]['population']
                        inv_truth_v3[i,] = inv_truth_v3[i,] / 10000 * state_pop[states[i]]['population']
                        # print(i+1,'/', len(inv_forecasts_v3))
                    # print(inv_forecasts_v3.shape, inv_truth_v3.shape)

                    forecasts_ql.append(inv_forecasts_v3)

            if nth_week == 1:
                forecasts_ql_4wk = forecasts_ql
                inv_truth_v3_4wk = inv_truth_v3
                mae_list_v3_4wk = mae_list_v3
            else:
                ## Concatenate 4 Weeks' forecasts
                inv_truth_v3_4wk = np.concatenate((inv_truth_v3_4wk,
                                                   inv_truth_v3), axis=1)
                mae_list_v3_4wk = mae_list_v3_4wk + mae_list_v3

                for q in range(len(forecasts_ql)):
                    forecasts_ql_4wk[q] = np.concatenate((forecasts_ql_4wk[q],
                                                          forecasts_ql[q]), axis=1)

            # new_row = [datetime(2021,11,8) + timedelta(days=i)] + mae_list_v3
            # mae_df.loc[len(mae_df)] = new_row
            # mae_df.to_csv('mae_1st_week_cont.csv')

        # ## Export i-th run to CSV file
        export_forecasts(inv_forecast=forecasts_ql_4wk, date=train_date,
                         n_seq=28, states_list=states, train_date=train_date)
