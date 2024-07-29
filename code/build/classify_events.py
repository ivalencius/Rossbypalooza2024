import xarray as xr
import numpy as np
import metpy
from tqdm import tqdm
import os
from rich import print
import warnings

# Datasets
ERA5 = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr',)

def wrapped_gradient(da, coord):
    """Finds the gradient along a given dimension of a dataarray."""

    dims_of_coord = da.coords[coord].dims
    if len(dims_of_coord) == 1:
        dim = dims_of_coord[0]
    else:
        raise ValueError('Coordinate ' + coord + ' has multiple dimensions: ' + str(dims_of_coord))
 
    coord_vals = da.coords[coord].values
    return xr.apply_ufunc(np.gradient, da, coord_vals, kwargs={'axis': -1},
                      input_core_dims=[[dim], []], output_core_dims=[[dim]],
                      output_dtypes=[da.dtype])
    
def load_IBTrACS():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        IBTrACS = xr.open_dataset('../../data/IBTrACS.since1980.v04r01.nc')
        all_times = IBTrACS.time.values.flatten()
        # reshaping (storm, time) introduces lots of nan times (get set to 1858ish)
        time_mask = all_times >= np.datetime64('1980-01-01')
    times_with_events = all_times[time_mask].astype('datetime64[ns]')
    lats_with_events = IBTrACS.lat.values.flatten()[time_mask]
    lons_with_events = IBTrACS.lon.values.flatten()[time_mask]
    return times_with_events, lats_with_events, lons_with_events

class PrecipEvent():
    def __init__(
        self, 
        id,
        start_date, end_date,
        start_centroid, end_centroid, 
        total_precip, avg_precip_per_day
        ):
        """Initialize a PrecipEvent

        Args:
            id (int): Label of the event.
            start_date (pd.Datetime): Start day of event.
            end_date (pd.Datetime): End day of event.
            start_centroid (tuple): (lon, lat) of the start centroid.
            end_centroid (tuple): (lon, lat)
            total_precip (float64): Total precipitation (units ???)
            avg_precip_per_day (float64): Total precipitation divided by duration.

        Returns:
            None: None
        """
        self.id = id
        self.start_date = np.datetime64(start_date)
        self.end_date = np.datetime64(end_date)
        # If start and end date are the same, then the event is a single day
        self.duration = (start_date - end_date).astype('timedelta64[D]') + 1
        self.total_precip = total_precip
        self.avg_precip_per_day = avg_precip_per_day
        self.start_centroid = start_centroid
        self.end_centroid = end_centroid
        self.bounding_box_5deg = [
            start_centroid[0] - 2.5, # min lon
            start_centroid[0] + 2.5, # max lon
            np.min([start_centroid[1] - 2.5, start_centroid[1] + 2.5]), # min lat
            np.max([start_centroid[1] - 2.5, start_centroid[1] + 2.5]), # max lat
        ]
        return None
    
    def __repr__(self):
        """
        Returns a string formatted for printing power plant data.

        Args:
            None

        Returns:
            String.
        """
        attrs = vars(self)
        return ',\n'.join("\t[bold black]%s[/bold black]: %s" % item for item in attrs.items())
    
    def add_ERA5(self):
        file = f'../../data/precip-events-ERA5/{self.id}.nc'
        if os.path.exists(file):
            ERA5_subset = xr.open_dataset(file)
        else:
            # Get 10 degree bounding box from (lat, lon) centroid
            start_buffer = self.start_date - np.timedelta64(2, 'D')
            end_buffer = self.end_date + np.timedelta64(2, 'D')
            # Need to change latitude slice based on hemisphere
            unsaved = ERA5[[
                    'total_precipitation_6hr',
                    'boundary_layer_height',
                    # All nan for some reason
                    # 'mean_vertically_integrated_moisture_divergence',
                    'integrated_vapor_transport',
                    'temperature',
                    'geopotential',
                    'u_component_of_wind',
                    'v_component_of_wind',
                    'wind_speed',
                    'vertical_velocity',
                    'vorticity',
                    'specific_humidity',
                    
                ]].sel(
                time=slice(start_buffer, end_buffer),
                longitude=slice(self.bounding_box_5deg[0], self.bounding_box_5deg[1]),
                # Need to do max lat first
                latitude=slice(self.bounding_box_5deg[3], self.bounding_box_5deg[2]),
                # level=slice(600, 1000),
            )
            unsaved.to_netcdf(file)
        ERA5_subset = xr.open_dataset(file)
        self.ERA5 = ERA5_subset
    
    def __check_for_TC(self):
        # 1 - check if there is event in same time period
        times_with_events, lats_with_events, lons_with_events = load_IBTrACS()
        day_difference = np.abs(
            (times_with_events - self.start_date).astype('timedelta64[D]')
            )
        min_day_difference = np.min(day_difference)
        if not min_day_difference <= np.timedelta64(1, 'D'):
            return False
        # 2 - check if event is within 5 degrees
        index_IBTrACS = np.argwhere(day_difference == min_day_difference)
        # Check all days with minimum day difference
        for day_idx in index_IBTrACS:
            lon_IBTrACS  = lons_with_events[day_idx]
            lat_IBTrACS = lats_with_events[day_idx]
            # Need to deal with longitude formating (want 0 -> 360)
            if lon_IBTrACS < 0:
                lon_IBTrACS += 360
            good_lon = (lon_IBTrACS >= self.bounding_box_5deg[0]) and (lon_IBTrACS <= self.bounding_box_5deg[1])
            # Condition depends if latitude is negative
            if self.bounding_box_5deg[2] > 0: # Positve latitude
                good_lat = (lat_IBTrACS >= self.bounding_box_5deg[2]) and (lat_IBTrACS <= self.bounding_box_5deg[3])
            else: # Negative latitude
                good_lat = (lat_IBTrACS <= self.bounding_box_5deg[3]) and (lat_IBTrACS <= self.bounding_box_5deg[2])
            if good_lon and good_lat:
                return True
        return False
    
    def __check_for_SurfaceFronts(self):
        # Departure from 900 hPa
        temperature = self.ERA5.temperature.sel(level=925).mean('time')
        vorticity = self.ERA5.vorticity.sel(level=925).mean('time')
        lon_grad = wrapped_gradient(temperature, 'longitude')/111 # degree -> km
        lat_grad = wrapped_gradient(temperature, 'latitude')/111
        grads = np.sqrt(lon_grad**2 + lat_grad**2)
        F_param = grads * vorticity
        scale = 0.45/100 # 0.45 K/100 km
        coriolis = metpy.calc.coriolis_parameter(temperature.latitude).values
        F_star = (F_param / (scale*coriolis)).to_dataset(name='F_star').F_star
        F_star.to_netcdf(f'../../data/F-star/{self.id}.nc')
        # # Check if a decent amount of F-star is above 1
        # indicator_proportion = np.sum(F_star >= 1)/F_star.size
        # if indicator_proportion >= 0.2:
        #     return True
        # else:
        #     return False
        # From Smirnov et al. (2015) two or more neighbor gridpoints must be masked to be a front
        F_star = F_star.values
        front_cells = F_star.where(F_star >= 1)
        # Pad array to ensure no edge effects
        padded = np.pad(front_cells, 1, mode='constant', constant_values=0)
        # Check if any cells neighbor each other
        for i in range(front_cells.shape[0]-1):
            for j in range(front_cells.shape[1]-1):
                if front_cells[i,j] == 1:
                    has_neighbor = (
                        padded[i+1, j] == 1 or
                        padded[i+1, j+1] == 1 or
                        padded[i, j+1] == 1 or
                        padded[i-1, j+1] == 1 or 
                        padded[i-1, j] == 1 or 
                        padded[i-1, j-1] == 1 or 
                        padded[i, j-1] == 1 or
                        padded[i+1, j-1] == 1
                    )
                    if has_neighbor:
                        return True
        return False
        
    def __check_for_Thunderstorm(self):
        Cp = 1005 # Specific heat of air [J/kgK]
        L = 2.5e6 # Latent heat of vaporization of liquid water [J/k]
        # g = 9.81 # Gravitational acceleration [m/s2]
        right_before = self.start_date - np.timedelta64(1, 'D')
        T = self.ERA5.temperature.sel(time=right_before)
        q = self.ERA5.specific_humidity.sel(time=right_before)/1000 # kg/kg to g/kg
        z = self.ERA5.geopotential.sel(time=right_before) # This is already multiplied by g
        # Now need to determine saturation specific humidity (Classius-Clapyeron -> q_star)
        # Step 1. Saturation vapor pressure
        # https://cran.r-project.org/web/packages/humidity/vignettes/humidity-measures.html
        e_s = 6.11*np.exp((L/461.52)*((1/273.15)-(1/T.sel(level=500))))
        # Step 2. Saturation specific humidity (at 500 hPa)
        q_star = (e_s*0.622)/(500-(1-0.622)*e_s)
        # Get moist static energy (coerce to kJ/kg)
        MSE = (Cp*T.sel(level=1000) + L*q.sel(level=1000) + z.sel(level=1000))/1000
        MSE_star = (Cp*T.sel(level=500) + L*q_star + z.sel(level=500))/1000
        # Get CAPE proxy
        CAPE_p = xr.combine_by_coords([MSE.to_dataset(name='MSE').drop_vars('level'), MSE_star.to_dataset(name='MSE_star').drop_vars('level')])
        CAPE_p['CAPE_proxy'] = (CAPE_p.MSE - CAPE_p.MSE_star)
        CAPE_p.to_netcdf(f'../../data/CAPE-proxy/{self.id}.nc')
        # Get maximum CAPE of that day
        # High CAPE threshold from Tuckman et. al (2022)
        indicator_proportion = np.sum(CAPE_p.CAPE_proxy >= 1.75)/CAPE_p.CAPE_proxy.size
        if indicator_proportion >= 0.2:
            return True
        else:
            return False
        
        
        
    def add_event(self):
        self.is_TC = self.__check_for_TC()
        self.is_SurfaceFront = self.__check_for_SurfaceFronts()
        self.is_Thunderstorm = self.__check_for_Thunderstorm()
        return None
    
    
if __name__ == '__main__':
    event_data = xr.open_dataset('../../data/Ext_Precip_999.nc')
    for e in tqdm(sorted(event_data.event.values), desc='Processing events'):
        ds = event_data.sel(event=e)
        event = PrecipEvent(
            e,
            ds.start_date.values, ds.end_date.values,
            (ds.start_lon.item(), ds.start_lat.item()), (ds.end_lon.item(), ds.end_lat.item()),
            ds.TPV.values,
            ds.PPD.values,
        )
        event.add_ERA5()
        # event.add_event()
        # print(repr(event))
        