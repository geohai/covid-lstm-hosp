import pandas as pd
import numpy as np
import glob
from scipy.stats import gamma, rv_continuous, t
import warnings
warnings.filterwarnings("ignore")

# input forecast date to access folder (String)
forecast_date = '2022-08-06'
folder_name = forecast_date+'-CUBoulder-RS-LSTM'
path_list = glob.glob(f"./{folder_name}/*.csv")
# path_list = glob.glob(f"./omicron_evaluation_15month(non-sph)/{folder_name}/*.csv")
n_files = len(path_list)
print(len(path_list))

all_df = pd.concat([pd.read_csv(path) for path in path_list])

# Get all target_end_date, quantiles and states_id
date_list = all_df['target_end_date'].unique().tolist()
quantile_list = sorted(all_df['quantile'].unique().tolist())
state_list = all_df['location'].unique().tolist()

# Create new dataframe to store median forecasts
median_df = pd.DataFrame()

for date in date_list:
    print(f'========= Date Process: {date_list.index(date) + 1}/{len(date_list)} =========')
    for state in state_list:
        print(f'Processing State: {state :02d} || {state_list.index(state) + 1}/{len(state_list)}')
        median_date_state_df = pd.DataFrame()
        for quantile in quantile_list:
            # print(f'Processing Quantile: {quantile}')
            df_quantile_state = all_df[(all_df['quantile'] == quantile) &
                                       (all_df['location'] == state) &
                                       (all_df['target_end_date'] == date)]
            # Median or Mean???
            median_row = df_quantile_state.loc[df_quantile_state['value'] == df_quantile_state['value'].median()]
            ########### Mean ###########
            # mean_value = df_quantile_state['value'].mean()
            # median_row.loc[:, ('value')] = mean_value
            median_date_state_df = median_date_state_df.append(median_row)
        # print(median_date_state_df)

        # Check Order of Quantile
        if median_date_state_df['value'].tolist() != sorted(sorted(median_date_state_df['value'])):
            print('Order of quantiles needs to be changed')
            #             print(median_date_state_df['value'].tolist(), '\n',
            #                   sorted(median_date_state_df['value']))
            # Check Negativity of forecasts
            if sum(n < 0 for n in median_date_state_df['value'].tolist()) > 0:
                print('Have negative values')
            values = np.array(sorted(median_date_state_df['value']))
            values[values < 0] = 0
            median_date_state_df['value'] = values

        median_df = median_df.append(median_date_state_df)

#####################        Don't Use This                #####################
##################### Calculating national forecasts (SUM) #####################
#####################           Not Right                  #####################
# print('Processing national forecasts')
# us_median_df = median_df.groupby(['quantile', 'target_end_date', 'target']).sum().reset_index()
# for date in date_list:
#     us_median_date_df = us_median_df[(us_median_df['target_end_date']==date)][:-1]
#     # if order of quantiles
#     if us_median_date_df['value'].tolist() != sorted(us_median_date_df['value']):
#         print('Order of quantiles needs to be changed')
#         us_median_date_df['value'] = sorted(us_median_date_df['value'])
# us_median_df['location'] = 'US'
# us_median_df['type'] = 'quantile'
# us_median_df['forecast_date'] = median_df['forecast_date'][0]

#####################            Use This                  #####################
##################### Calculating national forecasts (SUM) #####################
us_median_df = pd.DataFrame()

for date in date_list:
    print(f'========= Date Process: {date_list.index(date) + 1}/{len(date_list)} =========')
    std_list = []
    mean_list = []
    sample_size = 320
    for state in state_list:
        print(f'Processing State: {state :02d} || {state_list.index(state) + 1}/{len(state_list)}')

        date_state_dist = median_df[(median_df['target_end_date'] == date) &
                                    (median_df['location'] == state)]['value'].tolist()

        a, loc, scale = gamma.fit(date_state_dist)
        mean = date_state_dist[int((len(date_state_dist) - 1) / 2)]
        mean_list += [mean]
        std = gamma.std(a, loc, scale)
        std_list += [std]
    national_std = np.sqrt(sum(i * i for i in std_list))
    national_mean = sum(mean_list)
    print('mean: ', national_mean, ', std: ', national_std)
    for quantile in quantile_list:
        if quantile == 0.5:
            us_row = median_df[(median_df['target_end_date'] == date) &
                               (median_df['location'] == state) &
                               (median_df['quantile'] == quantile)]
            us_row['location'] = 'US'
            # us_row['quantile'] = 0.500
            us_row['value'] = national_mean
        elif quantile < 0.5:
            CI = t.interval(alpha=1 - quantile, df=sample_size - 1, loc=national_mean, scale=national_std)[0]
            if CI < 0:
                CI = 0
            us_row = median_df[(median_df['target_end_date'] == date) &
                               (median_df['location'] == state) &
                               (median_df['quantile'] == quantile)]
            us_row['location'] = 'US'
            # us_row['quantile'] = quantile
            us_row['value'] = CI
        else:
            CI = t.interval(alpha=quantile, df=sample_size - 1, loc=national_mean, scale=national_std)[-1]
            if CI < 0:
                CI = 0
            us_row = median_df[(median_df['target_end_date'] == date) &
                               (median_df['location'] == state) &
                               (median_df['quantile'] == quantile)]
            us_row['location'] = 'US'
            # us_row['quantile'] = quantile
            us_row['value'] = CI

        # append to Dataframe
        us_median_df = us_median_df.append(us_row)

## Concatenate national forecasts
median_df = median_df.append(us_median_df)

## can not use = to get nan
## replicate 0.500 quantile for point
point_df = median_df[median_df['quantile'] == 0.5]
point_df.loc[:, 'type'] = 'point'
point_df.loc[:, 'quantile'] = 'NA'
median_df = median_df.append(point_df)

median_df['location'] = median_df['location'].astype(str).str.pad(2, side='left', fillchar='0')

# Double-check Negative value issue
median_df.loc[median_df['value'] < 0, 'value'] = 0
median_df.to_csv(folder_name + '_median.csv', index=False)
# median_df.to_csv(f"./omicron_evaluation_15month(non-sph)/{folder_name}_median.csv", index=False)
print('Exporting to CSV finished!')