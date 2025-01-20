import pandas as pd
from utils import get_control_clean
from isc_lib import isc
import warnings
warnings.filterwarnings("ignore")


def get_intertisial_data(data_path, target_var):
    data = pd.read_csv(data_path, index_col=0)
    
    treated = data[data['treatment_group']==True].copy()
    controls = data[data['treatment_group']==False].copy()

    target_var = target_var
    samples = get_control_clean(controls, treated,
                                [target_var,
                                'dvage',
                                'mastat_recoded',
                                'sex_recoded',
                                'employed_num',
                                'hhsize',
                                #'asian',
                                #'black',
                                #'mixed',
                                #'other',
                                'low',
                                'middle'],
                                target_var,
                                'weight_yearx')
    return samples


def run_isc(data_path, target_var, out_suffix, k_n=35):
    print(f'Getting data for {target_var}_{out_suffix}...')
    samples = get_intertisial_data(data_path, target_var)
    print('DONE')
    print(f'Running ISC for {target_var}_{out_suffix}...')
    out = isc(samples, penalized=True, reduction=True, k_n=k_n)
    print('DONE')
    print('Saving Data...')
    diffs = pd.concat(out['diffs'], axis=1).sort_index()
    w_diffs = pd.concat(out['w_diffs'], axis=1).sort_index()
    treats = pd.concat(out['treats'], axis=1).sort_index()
    w_treats = pd.concat(out['w_treats'], axis=1).sort_index()
    synths = pd.concat(out['synths'], axis=1).sort_index()
    w_synths = pd.concat(out['w_synths'], axis=1).sort_index()
    boots_vars = pd.concat(out['boots_vars'], axis=1).mean(axis=1).sort_index()
    diffs.to_csv(f'./outputs/diffs_{target_var}_{out_suffix}.csv')
    w_diffs.to_csv(f'./outputs/w_diffs_{target_var}_{out_suffix}.csv')
    treats.to_csv(f'./outputs/treats_{target_var}_{out_suffix}.csv')
    w_treats.to_csv(f'./outputs/w_treats_{target_var}_{out_suffix}.csv')
    synths.to_csv(f'./outputs/synths_{target_var}_{out_suffix}.csv')
    w_synths.to_csv(f'./outputs/w_synths_{target_var}_{out_suffix}.csv')
    boots_vars.to_csv(f'./outputs/boots_vars_{target_var}_{out_suffix}.csv')
    print('DONE')
    return 0


if __name__ == "__main__":
    run_isc('./data/bybg/ii_high_nw.csv', 'ind_inc_deflated', 'high_nw', 10)
    run_isc('./data/bybg/ii_low_nw.csv', 'ind_inc_deflated', 'low_nw', 10)
    run_isc('./data/bybg/ii_high_w.csv', 'ind_inc_deflated', 'high_w', 10)
    run_isc('./data/bybg/ii_low_w.csv', 'ind_inc_deflated', 'low_w', 10)

    run_isc('./data/bybg/hhi_high_nw.csv', 'hh_inc_deflated', 'high_nw', 10)
    run_isc('./data/bybg/hhi_low_nw.csv', 'hh_inc_deflated', 'low_nw', 10)
    run_isc('./data/bybg/hhi_high_w.csv', 'hh_inc_deflated', 'high_w', 10)
    run_isc('./data/bybg/hhi_low_w.csv', 'hh_inc_deflated', 'low_w', 10)
