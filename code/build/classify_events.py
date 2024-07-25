import xarray as xr
import pandas as pd
import numpy as np
import datetime

# Datasets
ERA5 = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')


def load_IBTrACS():
    IBTrACS = xr.open_dataset('../../data/IBTrACS.since1980.v04r01.nc')
    all_times = IBTrACS.time.values.flatten()
    time_mask = all_times >= pd.to_datetime('1980-01-01')
    times_with_events = all_times[time_mask]
    lats_with_events = IBTrACS.lat.values.flatten()[time_mask]
    lons_with_events = IBTrACS.lon.values.flatten()[time_mask]
    return times_with_events, lats_with_events, lons_with_events, IBTrACS
    


def get_ERA5(id, bounding_box, start_date, end_date):
    file = f'../../precip-event-ERA5/{id}.nc'
    try:
        ERA5_subset = xr.open_dataset(file)
    except:
        # Get 10 degree bounding box from (lat, lon) centroid
        start_buffer = start_date - datetime.timedelta(days=2)
        end_buffer = end_date + datetime.timedelta(days=2)
        unsaved = ERA5[[
                'boundary_layer_height',
                'mean_vertically_integrated_moisture_divergence',
                'temperature',
                'u_component_of_wind',
                'v_component_of_wind',
                'vorticity',
                'winds_speed',
                
            ]].sel(
            longitude=slice(bounding_box[0], bounding_box[1]),
            latitude=slice(bounding_box[2], bounding_box[3]),
            level=slice(600, 1000),
            time=slice(start_buffer, end_buffer),
        )
        unsaved.to_netcdf(file)
        ERA5_subset = xr.open_dataset(file)
    return ERA5_subset

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
        self.duration = (start_date - end_date).astype('timedelta64[day]')
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
        self.ERA5 = get_ERA5(id, self.bounding_box_5deg start_date, end_date)
        self.event = None
        return None
    
    def check_for_TC(self):
        # 1 - check if there is event in same time period
        times_with_events, lats_with_events, lons_with_events, IBTrACS = load_IBTrACS()
        day_difference = np.abs(
            (times_with_events - np.array()).astype('timedelta64[D]')
            )
        min_day_difference = np.min(day_difference)
        if min_day_difference != 1:
            return None
        # 2 - check if event is within 5 degrees
        index_IBTrACS = np.argmin(day_difference)
        lon_IBTrACS  = lons_with_events[index_IBTrACS]
        lat_IBTrACS = lats_with_events[index_IBTrACS]
        # Need to deal with latitude formating (want 0 -> 360)
        if lat_IBTrACS < 0:
            lat_IBTrACS += 360
        good_lon = (lon_IBTrACS >= self.bounding_box_5deg[0]) and (lon_IBTrACS <= self.bounding_box_5deg[1])
        # Condition depends if latitude is negative
        if self.bounding_box_5deg[2] > 0: # Positve latitude
            good_lat = (lat_IBTrACS >= self.bounding_box_5deg[2]) and (lat_IBTrACS <= self.bounding_box_5deg[3])
        else: # Negative latitude
            good_lat = (lat_IBTrACS <= self.bounding_box_5deg[2]) and (lat_IBTrACS <= self.bounding_box_5deg[3])
        if good_lon and good_lat:
            return True
        
    def add_event(self):
        self.is_TC = self.check_for_TC()
        self.is_Frontal = self.check_for_Fronts()
        self.is_Thunderstorm = self.check_for_Thunderstorm()
    
    
if __name__ == '__main__':
    pass