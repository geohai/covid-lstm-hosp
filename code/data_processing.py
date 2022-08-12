from math import sqrt
import numpy as np
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from operator import add
from datetime import datetime, timedelta


## Loading hospitalization data
hosp_link = "https://github.com/reichlab/covid19-forecast-hub/raw/master/data-truth/truth-Incident%20Hospitalizations.csv"
df = read_csv(hosp_link)
df['date'] = pd.to_datetime(df['date'])
print(f'Earlies Date: {min(df["date"])}')
print(f'Latest Date: {max(df["date"])}')

states = list(df['location_name'].unique())
states.remove('United States')
states.remove('Virgin Islands')
states.remove('American Samoa')
# states.remove('Hawaii')
states.remove('Puerto Rico')
print(f'Number of States (No U.S.): {len(states)}')

# As North Dakota's ealiest date of hospitalization data
# available is 2020-07-27
# All the data will truncate with this date
df = df[df['date'] >= pd.Timestamp('2020-07-27')]
df = df[df['location_name'].isin(states)]

print(f"Each State has {int(len(df)/len(states))} days data available.\
        \nTotal states is {len(states)}")


## Loading Cases Data
case_link = "https://github.com/reichlab/covid19-forecast-hub/blob/master/data-truth/truth-Incident%20Cases.csv?raw=true"
case_df = read_csv(case_link, low_memory=False)
case_df['date'] = pd.to_datetime(case_df['date'])
# Align Date
case_df = case_df[(case_df['date'] >= pd.Timestamp('2020-07-27')) & \
                  (case_df['date'] <= max(df['date']))]
# Filter States
case_df = case_df[case_df['location_name'].isin(states)]

print(f"Each State has {int(len(case_df)/len(states))} days data available.\
        \nTotal states is {len(states)}")

case_df.loc[case_df['value']<0, 'value'] = 0

## Loading Population Data
filename = '../data/locations.csv'
pop_df = pd.read_csv(filename)

pop_df = DataFrame(pop_df)
pop_df = pop_df[-pop_df['abbreviation'].isna()]
pop_df = pop_df[['location_name', 'population']]
pop_df = pop_df.set_index('location_name')
# pop_df = pop_df.groupby('State_Name').sum()

pop_dict = pop_df['population'].to_dict()
pop_df.to_csv('state_pop_hub.csv')

## Smooth the data with a 14-day window
## You can change this based on the reporting frequency
smoothed_hosp_df = pd.DataFrame()
smoothed_case_df = pd.DataFrame()

for state in states:
    temp_df = df[df['location_name'] == state].sort_values(by='date')
    temp_df['smoothed_value'] = temp_df.rolling(window=14).mean()
    smoothed_hosp_df = smoothed_hosp_df.append(temp_df)

    state_case = case_df[case_df['location_name'] == state].sort_values(by='date')
    state_case['smoothed_value'] = state_case.rolling(window=14).mean()
    smoothed_case_df = smoothed_case_df.append(state_case)

len_ = len(smoothed_hosp_df)
smoothed_hosp_df = smoothed_hosp_df.dropna(subset=['smoothed_value'])
smoothed_case_df = smoothed_case_df.dropna(subset=['smoothed_value'])

print('Actual drop = ',len_ - len(smoothed_hosp_df), ', Num of states * 6 =',
      len(states)*6)
print('Actual drop = ',len_ - len(smoothed_case_df), ', Num of states * 6 =',
      len(states)*6)
print(len(smoothed_hosp_df)==len(smoothed_case_df))

main_df = smoothed_hosp_df.copy()
main_df = main_df.rename(columns={'smoothed_value': 'smoothed_hosp',
                                  'value': 'hospitalizations'})
main_df = pd.merge(main_df, smoothed_case_df, how='inner',
                   on=['date', 'location', 'location_name'])

main_df = main_df.rename(columns={'value': 'cases',
                                  'smoothed_value': 'smoothed_cases'})

## Calculate Spatial Proximity to Cases (SPC)
## and Spatial Proximity to Hospitalizations (SPH)

sci_df = pd.read_csv('../data/gadm1_nuts2-gadm1_nuts2-fb-social-connectedness-index-october-2021/gadm1_nuts2_gadm1_nuts2.tsv',
                  delimiter="\t", low_memory=False)

temp1_idx = sci_df[sci_df['user_loc'].str.contains('USA')].index
temp2_idx = sci_df[sci_df['fr_loc'].str.contains('USA')].index
idx = temp1_idx.intersection(temp2_idx)

sci_df = sci_df.loc[idx]

gadm_us = read_csv('gadm1_nuts2-gadm1_nuts2-fb-social-connectedness-index-october-2021/gadm_us_states.csv')
gadm_us = gadm_us.set_index('gadm_id')
gadm_us_dict = gadm_us['Name'].to_dict()

inv_gadm_us_dict = {v: k for k, v in gadm_us_dict.items()}

sci_df = sci_df[(sci_df['user_loc'].isin(gadm_us_dict))
                &(sci_df['fr_loc'].isin(gadm_us_dict))]

# Calculate fixed weights
sci_weight_dict = {}

for gadm_id in gadm_us_dict:
    temp_df = sci_df[(sci_df['user_loc'] == gadm_id) &
                     (sci_df['fr_loc'] != gadm_id)]

    temp_df['weight'] = temp_df['scaled_sci'] / temp_df['scaled_sci'].sum()
    temp_df = temp_df.set_index('fr_loc')
    sci_weight_dict[gadm_id] = temp_df['weight'].to_dict()
    print(f"{gadm_id.replace('USA', '')} / {len(gadm_us_dict)}")

new_main_df = DataFrame()

for state in states:
    temp_df = main_df[main_df['location_name'] == state]

    i = 1
    for gadm_id in sci_weight_dict[inv_gadm_us_dict[state]]:
        if i == 1:
            temp_df2 = main_df[main_df['location_name'] == gadm_us_dict[gadm_id]]
            spc_temp = list(temp_df2['smoothed_cases'] / pop_dict[gadm_us_dict[gadm_id]] \
                            * 10000 * sci_weight_dict[inv_gadm_us_dict[state]][gadm_id])
            sph_temp = list(temp_df2['smoothed_hosp'] / pop_dict[gadm_us_dict[gadm_id]] \
                            * 10000 * sci_weight_dict[inv_gadm_us_dict[state]][gadm_id])
        else:
            temp_df2 = main_df[main_df['location_name'] == gadm_us_dict[gadm_id]]
            spc_temp = list(map(add, spc_temp,
                                list(list(temp_df2['smoothed_cases'] / pop_dict[gadm_us_dict[gadm_id]] \
                                          * 10000 * sci_weight_dict[inv_gadm_us_dict[state]][gadm_id]))))
            sph_temp = list(map(add, sph_temp,
                                list(list(temp_df2['smoothed_hosp'] / pop_dict[gadm_us_dict[gadm_id]] \
                                          * 10000 * sci_weight_dict[inv_gadm_us_dict[state]][gadm_id]))))
        i += 1
    temp_df['spc'] = spc_temp
    temp_df['sph'] = sph_temp

    new_main_df = new_main_df.append(temp_df)

    print(states.index(state) + 1, '/', len(states), ':', state)

## Export the processed data
new_main_df.to_csv(f'../processed_data/processed_df_{str(max(df["date"]).date())}.csv', index=False)