import xarray as xr
# Multi-threading
from dask.distributed import Client, LocalCluster
# Cosmetic options
from rich import print

SAVE_PATH = '/scratch/midway2/pjt5/'

def load_datasets():
    print('Loading datasets...')
    base_path = '/scratch/midway3/nnn/'
    open_kwargs = dict(
        chunks={'time':-1, 'lat':10, 'lon':10},
        parallel=True,
        # engine='h5netcdf',
    )
    ds_cp = xr.open_mfdataset(f'{base_path}*_cp.nc', **open_kwargs)
    ds_lsp = xr.open_mfdataset(f'{base_path}*_lsp.nc', **open_kwargs)
    # ds_cape = xr.open_mfdataset(f'{base_path}*_cape.nc', **open_kwargs)
    # Merge datasets
    ds = xr.merge([ds_cp, ds_lsp])
    return ds


def detrend(ds):
    # To obtain monthly means, we need to sum by day of year
    monthly_means = (
        ds.daily_tp
        .groupby('time.month').mean() # get means across months
    )
    monthly_means.isel(latitude=10, longitude=10).plot()
    # Get long term trends
    linear_fit = (
        ds.daily_tp
        .polyfit(dim='time', deg=1)
    )
    fit_values = xr.polyval(ds['time'], linear_fit.polyfit_coefficients)
    detrended_tp = ds.daily_tp.groupby('time.month') - monthly_means - fit_values
    return detrended_tp



if __name__ == "__main__":
    # Start cluster (must be single threaded for type safety)
    cluster = LocalCluster(n_workers=5, threads_per_worker=1)
    with Client(cluster) as client:
        print(f'Dask dashboard link: {client.dashboard_link}')
        # Do stuff...
        ds = load_datasets()
        # Edit dataset
        ds = (
            ds
            .assign(tp=lambda x: x.cp+x.lsp)
            # may remove one lat/lon coordinate if boundaries aren't divisible by 2
            .coarsen(latitude=2, longitude=2, boundary='trim').sum()
            .resample(time='1D').mean()
            .rename({'tp': 'daily_tp', 'cp': 'daily_cp', 'lsp': 'daily_lsp'})
        )
        detrended_tp = detrend(ds)
        # Get percentiles and normalized anomaly
        print('Obtaining percentiles...')
        tp_percentiles = detrended_tp.chunk({'time':-1}).quantile([0.9, 0.99, 0.999], dim='time')
        print('Obtaining normalized anomaly...')
        normalized_anomaly = (
            (detrended_tp - detrended_tp.mean('time')).groupby('time.month') / detrended_tp.groupby('time.month').std('time')
        )
        print('Saving new dataset...')
        ds_expanded = ds.assign(
                tp_percentiles=tp_percentiles,
                normalized_daily_tp_anomaly=normalized_anomaly
            )
        ds_expanded.to_netcdf(f'{SAVE_PATH}edited_daily_tp.nc')
        print('Done!')

        