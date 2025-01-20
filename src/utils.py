import glob
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def trimmer(x: pd.Series, lwb: float=0.0, upb: float=0.99):
    s = x.copy()
    lower_bound = s.quantile(lwb)
    upper_bound = s.quantile(upb)
    return s.apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)


def create_index(x):
    y = np.arange(len(x)) + 1
    return y - x

def create_relative_index(lst, point):
    index = lst.index(point)
    return [i - index for i in range(len(lst))]

def create_relative_MultiIndex(lst, point):
    index = lst.index(point)
    return [i - index for i in range(len(lst))]


def isc_data_preparation(data, conditions: dict):
    out = data.copy()
    employed = conditions['employed']
    dropna = conditions['dropna']
    target_var = conditions['target_var']
    min_treat_waves = conditions['min_treat_waves']
    min_waves_pretreat = conditions['min_waves_pretreat']
    if employed: # dropping if ever unemployed
        out['unemployed_bool'] = ~(out.employed == 1)
        out_copy = out.copy()
        to_drop = []
        for pid in out.pid.unique():
            temp_data = out[out.pid==pid].copy()
            if temp_data['unemployed_bool'].any(): #check if any time was unemployed
                to_drop.append(pid) # adds to drop if above is true
        out = out_copy[~out_copy.pid.isin(to_drop)].copy()
    if dropna: #dropping if ever missing in target var
        out_copy = out.copy()
        to_drop = []
        for pid in out.pid.unique():
            temp_data = out[out.pid==pid].copy()
            if temp_data[target_var].isnull().any():
                to_drop.append(pid)
        out = out_copy[~out_copy.pid.isin(to_drop)].copy()
    out['ever_treated'] = out.groupby('pid')['treated'].transform(any).values
    out['year_reindex'] = out.sort_values(by=['pid', 'year']).groupby('pid').cumcount() + 1
    out.reset_index(drop=True, inplace=True)
    out.sort_values(by=['pid', 'year_reindex'], inplace=True)
    out['year_treated'] = out.year[out.groupby('pid')['treated'].transform('idxmax').values].values
    out['year_treat_reindex'] = out.year_reindex[out.groupby('pid')['treated'].transform('idxmax').values].values
    out['initial_year'] = out.groupby('pid')['year'].transform('min').values
    out['reindex'] = out.groupby('pid')['year_treat_reindex'].transform(create_index).values
    out['years_treated'] = out.groupby('pid')['mother'].transform('sum').values
    treated = out[out.ever_treated].copy()
    control = out[(~out.ever_treated) & (out.mother==0)].copy()
    control['ever_treated'] = control.groupby('pid')['treated'].transform(any)
    treated = treated[~(treated.years_treated < min_treat_waves)]
    treated = treated.drop(treated[(treated.year_treat_reindex < min_waves_pretreat)].index)
    return treated, control

    
def get_control_clean(c_data, t_data, features, target_var, weights=None):
    samples = []
    t_ids = t_data.pid.unique().tolist()
    for t_id in t_ids:
        if t_data[t_data.pid == t_id].shape[0] < 5:
            continue
        out = {}
        treat_time = t_data[t_data.pid == t_id].year_treated.unique()[0]
        t_data = t_data.dropna(subset=['year']).copy()
        treat = t_data[t_data.pid == t_id].pivot(index='pid', columns='year')[features].T
        control = c_data.pivot(index='pid', columns='year')[features].T
        sub_sample = pd.concat([treat, control], axis=1, join="inner") # concat-join-inner ensure using index (year) as key
        out['data'] = sub_sample.dropna(axis=1).astype(np.float64)  # only complete columns
        out['treat_time'] = treat_time
        out['treat_id'] = t_id
        out['target_var'] = target_var
        #out['weight'] = t_data[t_data.pid == t_id][['year', weights]].set_index('year')
        samples.append(out)
    return samples


