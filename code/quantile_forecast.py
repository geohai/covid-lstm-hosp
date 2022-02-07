from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Concatenate
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
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
                           nth_week, n_features, pred_id=0):
    train = np.empty((0, n_lag * n_features + n_seq))
    test = np.empty((0, n_lag * n_features + n_seq))

    for state in states:
        state_df = series[series['location_name'] == state]
        state_df = state_df.drop(['location', 'location_name',
                                  'hospitalizations', 'cases', 'spc'], axis=1)

        state_supervised = series_to_supervised(state_df, n_in=n_lag, n_out=n_seq,
                                                n_features=n_features, pred_id=0)
        state_supervised_values = state_supervised.values
        state_train, state_test = state_supervised_values[0:-n_test], state_supervised_values[-n_test:]

        train = np.append(train, state_train, axis=0)
        test = np.append(test, state_test, axis=0)
        # all_state_supervised = all_state_supervised.append(state_supervised)
        # print(f'{state} is statcked in DataFrame.')

        # print(train.shape, test.shape)
    raw_values = np.concatenate((train, test), axis=0)
    print(f'Concatenated Dataframe shape: {raw_values.shape}')
    print(raw_values[1, -35:])
    if nth_week >= 2:
        raw_values = np.delete(raw_values, np.s_[n_features * n_lag: n_features * n_lag + 7 * (nth_week - 1)], axis=1)
    print(f'Concatenated Dataframe shape of {nth_week}th week: {raw_values.shape}')
    print(raw_values[1, -14:])
    # rescale values to -1, 1
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
def fit_lstm_quantile(train, n_lag, n_features,
                      n_seq, n_batch, nb_epoch,
                      n_neurons):
    # quantiles = [.025, .100, .250, .500, .750, .900, .975]
    quantiles = [.010, .025, .050, .100, .150, .200, .250, .300, .350, .400, .450, .500,
                 .550, .600, .650, .700, .750, .800, .850, .900, .950, .975, .990]
    output_dim = len(quantiles)
    print(output_dim)
    outputs = []

    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag * n_features], train[:, n_lag * n_features:]
    X = X.reshape(X.shape[0], n_lag, n_features)
    # design network
    ts_input = Input((X.shape[1], X.shape[2]))
    lstm_1_out = LSTM(96, activation="tanh", return_sequences=True)(ts_input)
    lstm_2_out = LSTM(96, activation="tanh")(lstm_1_out)
    # dense_1_out = Dense(96)(lstm_2_out)

    #     # create an output for each quantile
    #     for i,quantile in enumerate(quantiles):
    #         output = Dense(y.shape[1], name="q_"+str(i))(dense_1_out)
    #         outputs.append(output)
    outputs = [Dense(7, activation='tanh', name="q%d" % q_i)(lstm_2_out) for q_i in range(output_dim)]

    # loss
    q_loss = MultiQuantileLoss(quantiles)

    # intialize & compile
    model = Model(inputs=ts_input, outputs=outputs)
    model.compile(optimizer='adam', loss=q_loss.call)

    # model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(X, y, epochs=nb_epoch,
                        batch_size=n_batch,
                        verbose=2, shuffle=False)

    return model


# evaluate the persistence model
def make_forecasts(model, n_batch, test,
                   n_features, n_lag, n_seq):
    test_X, test_y = test[:, 0:n_lag * n_features], test[:, n_lag * n_features:]
    test_X = test_X.reshape((test_X.shape[0], n_lag, n_features))

    forecasts = model.predict(test_X, batch_size=n_batch)
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
def export_forecasts(inv_forecast, date, n_seq, states_list):
    # output name format = YYYY-MM-DD-team-model.csv
    output_name = date.strftime('%Y-%m-%d') + '-' + 'CUBoulder' + '-' + 'S_LSTM.csv'

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
                target = f'{j+n_seq-6} day ahead inc hosp'
                target_end_date = (date + timedelta(days=j+1)).strftime('%Y-%m-%d')
                location_fip = state_fip[states_list[i]]
                if quantile == 0.500:
                    type = 'point'
                    quantile_exp = 'NA'
                else:
                    type = 'quantile'
                    quantile_exp = quantile
                value = forecasts_quantile[i][j]

                row_exp = [date.strftime('%Y-%m-%d'), target, target_end_date,
                           location_fip, type, quantile_exp, value]
                export_df.loc[len(export_df.index)] = row_exp

    if not os.path.exists("./results/"):
        os.makedirs('./results/')
    export_df.to_csv('./results/'+output_name, index=False)

    print(f'{output_name} is successfully exported!')


if __name__ == '__main__':
    print('===== Start Reading Processed Data ======')
    df = read_csv('./data/procecssed_df_v2.csv', header=0, parse_dates=['date'], index_col=0)
    states = list(df['location_name'].unique())
    state_pop = read_csv('./data/state_pop.csv', index_col=0)
    state_pop = state_pop.to_dict('index')

    # Convert Raw numbers to Rate
    rate_df = DataFrame()
    for state in states:
        state_df = df[df['location_name'] == state]
        state_df['smoothed_hosp'] = state_df['smoothed_hosp'] / state_pop[state]['POPULATION'] * 10000
        state_df['smoothed_cases'] = state_df['smoothed_cases'] / state_pop[state]['POPULATION'] * 10000

        rate_df = rate_df.append(state_df)
        # print(f'{state} has been converted to rate.')
    print('Converting Raw numbers to rates completed!')

    col_names_1 = ['date', 't+1_MAE', 't+2_MAE', 't+3_MAE', 't+4_MAE',
                   't+5_MAE', 't+6_MAE', 't+7_MAE']
    col_names_2 = ['date', 't+8_MAE', 't+9_MAE', 't+10_MAE', 't+11_MAE',
                   't+12_MAE', 't+13_MAE', 't+14_MAE']
    col_names_3 = ['date', 't+15_MAE', 't+16_MAE', 't+17_MAE', 't+18_MAE',
                   't+19_MAE', 't+20_MAE', 't+21_MAE']
    col_names_4 = ['date', 't+22_MAE', 't+23_MAE', 't+24_MAE', 't+25_MAE',
                   't+26_MAE', 't+27_MAE', 't+28_MAE']

    mae_df = DataFrame(columns=col_names_1)

    # Define num of run times (Reduce Stochasticity) #
    n_run = 1
    # Define the period of training data #
    train_date = datetime(2021, 11, 1)

    for n in range(n_run):
        # Loop for 1st to 4th week forecasts
        for nth_week in range(1, 2):
            print(f'======================== {n + 1} / 1 ========================')
            print(f'========= Current Date: {train_date + timedelta(days=nth_week*7)} =========')
            wf_df = rate_df[rate_df.index <= (train_date + timedelta(days=nth_week*7))]
            # Configure
            n_features = 3
            n_lag = 21
            n_out = 7
            n_test = 1
            # nth_week = 1
            n_seq = nth_week * 7

            # Prepare data
            scaler, train, test = prepare_all_state_data(states, wf_df, n_test=n_test,
                                                         n_lag=n_lag, n_seq=n_seq, nth_week=nth_week,
                                                         n_features=n_features, pred_id=0)

            # n_feature * n_lag + n_seq
            # 4         * 21    + 7     = 91
            print("=================================================")
            print('Train: %s, Test: %s' % (train.shape, test.shape))

            X, y = train[:, 0:n_lag * n_features], train[:, n_lag * n_features:]
            X = X.reshape(X.shape[0], 1, X.shape[1])

            model = fit_lstm_quantile(train, n_lag=n_lag, n_features=n_features,
                                      n_seq=n_out, n_batch=2 ** 6, nb_epoch=1, n_neurons=96)

            print('Model training completed!')
            print('Now making forecast!')
            forecasts = make_forecasts(model, n_batch=2 ** 6, test=test,
                                       n_features=n_features,
                                       n_lag=n_lag, n_seq=n_out)
            print('Forecasts shape: ', forecasts[0].shape)
            print('Inverse forecast to unscaled numbers!')
            # inverse transform forecasts and test
            forecasts_ql = []
            for j in range(len(forecasts)):
                print('quantile_n:', j)
                if j == len(forecasts) // 2:
                    inv_forecasts_v3, inv_truth_v3 = inverse_transform_all_state(test=test, forecasts=forecasts,
                                                                                 scaler=scaler, n_lag=n_lag,
                                                                                 n_features=n_features, n_seq=n_out,
                                                                                 quantile_i=j)

                    for i in range(len(inv_forecasts_v3)):
                        inv_forecasts_v3[i, ] = inv_forecasts_v3[i, ] / 10000 * state_pop[states[i]]['POPULATION']
                        inv_truth_v3[i, ] = inv_truth_v3[i, ] / 10000 * state_pop[states[i]]['POPULATION']
                        # print(i+1,'/', len(inv_forecasts_v3))
                    # print(inv_forecasts_v3.shape, inv_truth_v3.shape)

                    forecasts_ql.append(inv_forecasts_v3)

                    mae_list_v3 = evaluate_forecasts(test=inv_truth_v3, forecasts=inv_forecasts_v3,
                                                     n_lag=n_lag, n_seq=n_out)
                else:
                    print('Not Median')
                    inv_forecasts_v3, inv_truth_v3 = inverse_transform_all_state(test=test, forecasts=forecasts,
                                                                                 scaler=scaler, n_lag=n_lag,
                                                                                 n_features=n_features, n_seq=n_out,
                                                                                 quantile_i=j)
                    for i in range(len(inv_forecasts_v3)):
                        inv_forecasts_v3[i, ] = inv_forecasts_v3[i, ] / 10000 * state_pop[states[i]]['POPULATION']
                        inv_truth_v3[i, ] = inv_truth_v3[i, ] / 10000 * state_pop[states[i]]['POPULATION']
                        # print(i+1,'/', len(inv_forecasts_v3))
                    # print(inv_forecasts_v3.shape, inv_truth_v3.shape)

                    forecasts_ql.append(inv_forecasts_v3)

            new_row = [train_date + timedelta(days=nth_week*7)] + mae_list_v3
            mae_df.loc[len(mae_df)] = new_row
            # mae_df.to_csv('mae_1st_week_cont.csv')

            export_forecasts(inv_forecast=forecasts_ql, date=train_date,
                             n_seq=n_seq, states_list=states)
