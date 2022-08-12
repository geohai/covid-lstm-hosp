# This is an alternative script to download model-required data
# But the data sources are the same as COVID ForecastHub
from math import sqrt
import numpy as np
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from datetime import datetime, timedelta
from operator import add
from sodapy import Socrata

# Loading Hospitalization Data
us_state_to_abbrev = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    "American Samoa": "AS",
    "Guam": "GU",
    "Northern Mariana Islands": "MP",
    "Puerto Rico": "PR",
    "United States Minor Outlying Islands": "UM",
    "U.S. Virgin Islands": "VI",
}

# invert the dictionary
abbrev_to_us_state = dict(map(reversed, us_state_to_abbrev.items()))

# Unauthenticated client only works with public data sets. Note 'None'
# in place of application token, and no username or password:
client = Socrata("healthdata.gov",
                 app_token='XXXXXXXXXXXXXXXX',
                 username='xxxxx@xxxx.edu',
                 password='XXXXXXXXXX')

# Example authenticated client (needed for non-public datasets):
# client = Socrata(healthdata.gov,
#                  MyAppToken,
#                  userame="user@example.com",
#                  password="AFakePassword")

# First 2000 results, returned as JSON from API / converted to Python list of
# dictionaries by sodapy.
results = client.get("g62h-syeh", limit=55000)

# Convert to pandas DataFrame
results_df = pd.DataFrame.from_records(results)
print(results_df.shape)

hosp_df = results_df[['state', 'date',
                      'previous_day_admission_adult_covid_confirmed',
                      'previous_day_admission_pediatric_covid_confirmed']]
hosp_df['date'] = hosp_df['date'].apply(datetime.fromisoformat)
hosp_df['state'] = hosp_df['state'].map(abbrev_to_us_state)

hosp_df['previous_day_admission_adult_covid_confirmed'] = hosp_df['previous_day_admission_adult_covid_confirmed'].astype('float')
hosp_df['previous_day_admission_pediatric_covid_confirmed'] = hosp_df['previous_day_admission_pediatric_covid_confirmed'].astype('float')

hosp_df['value'] = hosp_df['previous_day_admission_adult_covid_confirmed'] + \
                   hosp_df['previous_day_admission_pediatric_covid_confirmed']

hosp_df['date'] = hosp_df['date'] - timedelta(days=1)

hosp_df = hosp_df.rename(columns={'state': 'location_name'})

print(f'Earlies Date: {min(hosp_df["date"])}')
print(f'Latest Date: {max(hosp_df["date"])}')

states = list(hosp_df['location_name'].unique())
# states.remove('United States')
# states.remove('Virgin Islands')
states.remove('U.S. Virgin Islands')
states.remove('American Samoa')
# states.remove('Hawaii')
states.remove('Puerto Rico')
print(f'Number of States (No U.S.): {len(states)}')

hosp_df = hosp_df.drop(columns=['previous_day_admission_adult_covid_confirmed',
                                'previous_day_admission_pediatric_covid_confirmed'])
# Loading Cases Data
case_link = 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
case_df = read_csv(case_link, low_memory=False)
case_df = case_df.drop(columns=['UID', 'iso2', 'iso3', 'code3',
                                'FIPS', 'Admin2', 'Country_Region',
                                'Lat', 'Long_', 'Combined_Key'])
case_df = case_df.groupby(['Province_State']).sum()
case_df = case_df.drop(index=set(case_df.index)-set(states))
case_df.index.names = ['date']
case_df_T = case_df.transpose()
case_df_T.index = pd.to_datetime(case_df_T.index)

case_df_T = case_df_T[(case_df_T.index >= pd.Timestamp('2020-07-26')) & \
                  (case_df_T.index <= max(hosp_df['date']))]
case_df_T = case_df_T.diff()
case_df_T = case_df_T[case_df_T.index >= pd.Timestamp('2020-07-27')]
case_df_T


case_df = DataFrame(columns=['date', 'location_name', 'value'])
for rowIndex, row in case_df_T.iterrows(): #iterate over rows
    for columnIndex, value in row.items():
        row = [rowIndex, columnIndex, value]
        case_df.loc[len(case_df.index)] = row


case_df['value'] = case_df['value'].astype('float')
# Filter States
case_df = case_df[case_df['location_name'].isin(states)]

print(f"Each State has {int(len(case_df)/len(states))} days data available.\
        \nTotal states is {len(states)}")


# Loading Population Data
filename = '../data/locations.csv'
pop_df = pd.read_csv(filename)

pop_df = DataFrame(pop_df)
pop_df = pop_df[-pop_df['abbreviation'].isna()]
pop_df = pop_df[['location_name', 'population']]
pop_df = pop_df.set_index('location_name')
# pop_df = pop_df.groupby('State_Name').sum()

pop_dict = pop_df['population'].to_dict()
pop_df.to_csv('state_pop_hub.csv')

# Smooth the data with a 14-day window
# Change the smoothing magnitude based on reporting frequency
smoothed_hosp_df = pd.DataFrame()
smoothed_case_df = pd.DataFrame()

for state in states:
    temp_df = hosp_df[hosp_df['location_name'] == state].sort_values(by='date')
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
                   on=['date', 'location_name'])

main_df = main_df.rename(columns={'value': 'cases',
                                  'smoothed_value': 'smoothed_cases'})

# Calculate Spatial Proximity to Cases (SPC) and Spatial Proximity to Hospitalizations (SPH)
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

## Calculate fixed weights
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

state_fips = {'Hawaii': '15',
 'Nebraska': '31',
 'Iowa': '19',
 'New Hampshire': '33',
 'District of Columbia': '11',
 'Kansas': '20',
 'New Mexico': '35',
 'Maine': '23',
 'New Jersey': '34',
 'California': '06',
 'Idaho': '16',
 'Oregon': '41',
 'Louisiana': '22',
 'Mississippi': '28',
 'Rhode Island': '44',
 'South Dakota': '46',
 'Nevada': '32',
 'Maryland': '24',
 'Utah': '49',
 'Montana': '30',
 'North Dakota': '38',
 'Alaska': '02',
 'Connecticut': '09',
 'Delaware': '10',
 'Wyoming': '56',
 'Kentucky': '21',
 'Tennessee': '47',
 'Oklahoma': '40',
 'Colorado': '08',
 'Massachusetts': '25',
 'Minnesota': '27',
 'Arkansas': '05',
 'Michigan': '26',
 'South Carolina': '45',
 'Vermont': '50',
 'Georgia': '13',
 'Alabama': '01',
 'Washington': '53',
 'Virginia': '51',
 'New York': '36',
 'North Carolina': '37',
 'Arizona': '04',
 'Texas': '48',
 'Pennsylvania': '42',
 'Wisconsin': '55',
 'Florida': '12',
 'Illinois': '17',
 'Missouri': '29',
 'Indiana': '18',
 'West Virginia': '54',
 'Ohio': '39'}

new_main_df['location'] = new_main_df['location_name'].map(state_fips)

# Reorder the column
new_main_df = new_main_df[['date','location',
                           'location_name',
                           'hospitalizations',
                           'smoothed_hosp','cases',
                           'smoothed_cases','spc','sph']]
new_main_df.to_csv(f'../processed_data/processed_df_{str(max(new_main_df.date).date())}_b.csv', index=False)
