import xarray as xr
import numpy as np
from tqdm import tqdm
import os
from rich import print
import warnings

# Datasets
ERA5 = xr.open_zarr(
    'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr',
    # chunks={'time': 100, 'latitude': -1, 'longitude': -1},
    )

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
            start_centroid[1] - 2.5, # min lat
            start_centroid[1] + 2.5, # max lat
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
        file = f'../../data/precip-event-ERA5/{self.id}.nc'
        if os.path.exists(file):
            ERA5_subset = xr.open_dataset(file)
        else:
            # Get 10 degree bounding box from (lat, lon) centroid
            start_buffer = self.start_date - np.timedelta64(2, 'D')
            end_buffer = self.end_date + np.timedelta64(2, 'D')
            # Need to change latitude slice based on hemisphere
            if self.bounding_box_5deg[2] > 0:
                lat_slice = slice(self.bounding_box_5deg[2], self.bounding_box_5deg[3])
            else:
                lat_slice = slice(self.bounding_box_5deg[3], self.bounding_box_5deg[2])
            unsaved = ERA5[[
                    'total_precipitation_6hr',
                    'boundary_layer_height',
                    # All nan for some reason
                    # 'mean_vertically_integrated_moisture_divergence',
                    'integrated_vapor_transport',
                    'temperature',
                    'u_component_of_wind',
                    'v_component_of_wind',
                    'wind_speed',
                    'vertical_velocity',
                    'vorticity',
                    'specific_humidity',
                    
                ]].sel(
                time=slice(start_buffer, end_buffer),
                longitude=slice(self.bounding_box_5deg[0], self.bounding_box_5deg[1]),
                latitude=lat_slice,
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
        
    def add_event(self):
        self.is_TC = self.__check_for_TC()
        # self.is_Frontal = self.check_for_Fronts()
        # self.is_Thunderstorm = self.check_for_Thunderstorm()
        return None
    
    
if __name__ == '__main__':
    event_data = xr.open_dataset('../../data/Ext_Precip_999.nc')
    for e in tqdm(event_data.event.values, desc='Processing events'):
        ds = event_data.sel(event=e)
        event = PrecipEvent(
            e,
            ds.start_date.values, ds.end_date.values,
            (ds.start_lon.item(), ds.start_lat.item()), (ds.end_lon.item(), ds.end_lat.item()),
            ds.TPV.values,
            ds.PPD.values,
        )
        event.add_ERA5()
        event.add_event()
        print(repr(event))
        break