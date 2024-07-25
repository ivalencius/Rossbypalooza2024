import xarray as xr
from rich import print
from glob import glob
from tqdm import tqdm

SAVE_PATH = '/scratch/midway2/valencig/'

def parse_to_daily():
    print('Parsing to daily data...')
    base_path = '/scratch/midway3/nnn/'
    cp_files = sorted(glob(f'{base_path}*_cp.nc'))
    lsp_files = sorted(glob(f'{base_path}*_lsp.nc'))
    nfiles = len(cp_files)
    # Select expver for normal ERA5
    for i, (cp_file, lsp_file) in enumerate(zip(cp_files, lsp_files)):
        print(f'\t{i} of {nfiles}')
        ds_cp = xr.load_dataset(cp_file)
        ds_lsp = xr.load_dataset(lsp_file)
        try: 
            ds_cp = ds_cp.sel(expver=1)
            ds_lsp = ds_lsp.sel(expver=1)
        except:
            pass
        ds = xr.merge([ds_cp, ds_lsp])
        ds = (
            ds
            .coarsen(latitude=4, longitude=4, boundary='trim').sum()
            .resample(time='1D').sum()
            .assign(tp=lambda x: x.cp+x.lsp)
            .rename({'tp': 'daily_tp', 'cp': 'daily_cp', 'lsp': 'daily_lsp'})    
        ).persist()
        ds.to_netcdf(f'{SAVE_PATH}/daily-precip/{i}.nc', encoding={
            'daily_cp': {'dtype': 'float32'},
            'daily_lsp': {'dtype': 'float32'},
            'daily_tp': {'dtype': 'float32'},
        })


def detrend(ds):
    # print('Detrending data...')
    # To obtain monthly means, we need to sum by day of year
    monthly_means = (
        ds.daily_tp
        .groupby('time.month').mean() # get means across months
    )
    # Get long term trends
    linear_fit = (
        ds.daily_tp
        .polyfit(dim='time', deg=1)
    )
    fit_values = xr.polyval(ds['time'], linear_fit.polyfit_coefficients)
    detrended_tp = ds.daily_tp.groupby('time.month') - monthly_means - fit_values
    return detrended_tp

def get_anomalies():
    print('Determining anomalies...')
    ds = xr.open_mfdataset(f'{SAVE_PATH}/daily-precip/*.nc', parallel=True)
    detrended_tp = detrend(ds)
    # Get percentiles and normalized anomaly
    tp_percentiles = detrended_tp.quantile([0.9, 0.95, 0.99, 0.999], dim='time')
    normalized_anomaly = (
        (detrended_tp - detrended_tp.mean('time')).groupby('time.month') / detrended_tp.groupby('time.month').std('time')
    )
    ds_expanded = ds.assign(
            tp_percentiles_alltime=tp_percentiles,
            normalized_daily_tp_anomaly=normalized_anomaly
        )
    ds_expanded.to_netcdf(f'{SAVE_PATH}total-daily-precip.nc', encoding={
        'daily_cp': {'dtype': 'float32'},
        'daily_lsp': {'dtype': 'float32'},
        'daily_tp': {'dtype': 'float32'},
    })

if __name__ == "__main__":
    parse_to_daily()
    get_anomalies()
    print('Done!')