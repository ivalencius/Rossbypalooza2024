#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:26:27 2024

@author: kaying
"""
## Precipitation Event Detection ##
import numpy as np
import glob
import xarray as xr
import os
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from skimage.measure import label, regionprops
import xarray as xr
directory = "/Users/kaying/Documents/ERA5/Precip/"
file_pattern = os.path.join(directory, '*.nc')
file_list = sorted(glob.glob(file_pattern))
ds = xr.open_mfdataset(file_list, combine='by_coords')
ds = ds.isel(time=slice(0, 20))
P = ds.precip
P = P.chunk({'time': -1})

ndays = P.sizes['time']
nlat = P.sizes['latitude']
nlon = P.sizes['longitude']
#%% Extract the extreme precip values as P_ext
percentile = 0.97
P_threshold = P.quantile(percentile)  # can be an absolute threshold, to isolate precip pattern for event detection
P_ext = xr.where(P > P_threshold, P, 0) # set points below threshold as 0
#%% 3D connected component labeling
P_ext_np = P_ext.values
P_ext_np[P_ext_np!=0]=255 # Create P_ext_np as a binary np.array for labeling to work

# Apply 3D connected component labeling, labels start from 1
num_labels = label(P_ext_np, connectivity=3)  # 3D connectivity

labels_da = xr.DataArray(num_labels, dims=['time', 'latitude', 'longitude'], coords={'time': P_ext.time.values, 'latitude': P_ext.latitude.values, 'longitude': P_ext.longitude.values})
print(f"Number of labels: {np.max(num_labels)}")
nlabels = np.max(labels_da.values).astype(int)

# Find the start & End day of the event
regions = regionprops(num_labels)
Event_day = np.zeros((nlabels,2))
Event_xspan = np.zeros((nlabels,2))
Event_yspan = np.zeros((nlabels,2))
n = 0
for region in regions:
    minr, minc, mind, maxr, maxc, maxd = region.bbox
    Event_day[n,:] = [minr,maxr-1] # start & end day of event (i.e. vertices of object in time dimension)
    Event_xspan[n,:] = [minc,maxc]
    n+=1
Event_day = Event_day.astype(int)
#%% Find the centroids of first/last appearance of each label
start_day_centroid = np.zeros((nlabels,2))
end_day_centroid = np.zeros((nlabels,2))
for n in np.arange(nlabels):
    d1 = int(Event_day[n,0]) # start day
    d2 = int(Event_day[n,1]) # end day
    
    num_labels_start = num_labels[d1]==n+1 # labels start from 1
    
    if np.sum(num_labels_start)==0: 
        print("First day Error: There is no label ", n+1, " in day ", d1)
        start_day_centroid[n] = 'NaN'
    else:
        c = center_of_mass(num_labels_start)
        start_day_centroid[n,0] = labels_da.latitude[int(c[0])].values
        start_day_centroid[n,1] = labels_da.longitude[int(c[1])].values
    
    num_labels_end = num_labels[d2]==n+1 # labels start from 1
    
    if np.sum(num_labels_end)==0: 
        print("Last day Error: There is no label ", n+1, " in day ", d2)
        end_day_centroid[n] = 'NaN'
    else:
        c = center_of_mass(num_labels_end)
        end_day_centroid[n,0] = labels_da.latitude[int(c[0])].values
        end_day_centroid[n,1] = labels_da.longitude[int(c[1])].values
#%% Total precipitation & Precip frequency per event
Precip = np.zeros((nlabels, nlat,nlon))  # Total precip per event
PF = np.zeros((nlabels, nlat,nlon))  # Precip frequency per event
TP = np.zeros((nlabels))
for n in np.arange(nlabels):
    d1 = int(Event_day[n,0]) # start day
    d2 = int(Event_day[n,1]) # end day
    
    for t in np.arange(d1,d2+1):
        labels_mask = num_labels[t]==n+1 # labels start from 1
        Precip[n][labels_mask] += P_ext_np[t][labels_mask]
        PF[n] += labels_mask.astype(int)
    TP[n] = np.sum(Precip[n])
#%% Creating the Events DataArray
Events = xr.Dataset(
    {
        'start_date': (['event'], labels_da.time.values[Event_day[:,0]]),
        'end_date': (['event'], labels_da.time.values[Event_day[:,1]]),
        'duration': (['event'], Event_day[:,1]-Event_day[:,0]+1),
        'start_lat': (['event'], start_day_centroid[:,0]),
        'start_lon': (['event'], start_day_centroid[:,1]),
        'end_lat': (['event'], end_day_centroid[:,0]),
        'end_lon': (['event'], end_day_centroid[:,1]),  
        'TPV': (['event'], TP),
        'PPD': (['event'], TP/(Event_day[:,1]-Event_day[:,0]+1)),
        'total_precip': (['event', 'lat', 'lon'], Precip),
        'precip_freq': (['event', 'lat', 'lon'], PF)
    }
)   

#%% Attri Name
                             
Events.attrs['title'] = 'Precipitation Event'
Events.attrs['description'] = (
    f'Precipitation Events with precipitation threshold equals {P_threshold.values} mm/day '
    f'({percentile*100}th percentile), detected from {str(P.time.values[0])[:10]} '
    f'to {str(P.time.values[-1])[:10]}'
)

Events.start_date.attrs['longname'] = 'Start date'
Events.end_date.attrs['longname'] = 'End date'
Events.duration.attrs['longname'] = 'Duration'
Events.start_lat.attrs['longname'] = 'Start latitude'
Events.start_lon.attrs['longname'] = 'Start longitude'
Events.end_lat.attrs['longname'] = 'End latitude'
Events.end_lon.attrs['longname'] = 'End longitude'
Events.TPV.attrs['longname'] = 'Total precipitation value'
Events.PPD.attrs['longname'] = 'Precipitation per day'
Events.total_precip.attrs['longname'] = 'Total precipitation'
Events.precip_freq.attrs['longname'] = 'Precipitation frequency'

Events.start_date.attrs['description'] = 'Start date of the event'
Events.end_date.attrs['description'] = 'End date of the event'
Events.duration.attrs['description'] = 'Duration of the event'
Events.start_lat.attrs['description'] = 'Latitude of centroid of the event on start date'
Events.start_lon.attrs['description'] = 'Longitude of centroid of the event on start date'
Events.end_lat.attrs['description'] = 'Latitude of centroid of the event on end date'
Events.end_lon.attrs['description'] = 'Longitude of centroid of the event on end date'
Events.TPV.attrs['description'] = 'Total precipitation value throughout the event'
Events.PPD.attrs['description'] = 'Average precipitation per day for the event'
Events.total_precip.attrs['description'] = '2D total precipitation pattern of the event'
Events.precip_freq.attrs['description'] = '2D precipitation frequency (number of day) pattern for the event'
#%% Sanity Check
# i = 7 # select a label (starting from 1)
# D1 = np.where(labels_da.time.values == Events['Start date'].isel(event=i-1).values)[0][0] # day index of start date of event, i-1 as index starts from 0
# D2 = np.where(labels_da.time.values == Events['End date'].isel(event=i-1).values)[0][0] # day index of start date of event, i-1 as index starts from 0
# N = Events.Duration.values[i]


# label_masked = labels_da == i
# for t in np.arange(D1, D2+1):
#     l = label_masked.isel(time=t).values!=0
#     if np.sum(l) ==0:
#         print('day ', t, ' has no label ',i)
#     else:
#         print('day ',t, ' has ',np.sum(l),' grid points')

# for t in np.arange(D1, D2+1):
#     plt.figure(figsize=(12, 7))
#     label_masked.isel(time=t).plot(cmap='binary')
#     x1 = Events['Start Longitude'].isel(event=i-1).values
#     y1 = Events['Start Latitude'].isel(event=i-1).values
#     x2 = Events['End Longitude'].isel(event=i-1).values
#     y2 = Events['End Latitude'].isel(event=i-1).values
#     # print(y1,x1,y2,x2)
#     plt.show()

# plt.figure()
# plt.contourf(Precip[i-1], cmap = 'Blues')
# plt.title('Total Precip for Event '+str(int(i)))
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.contourf(PF[i-1], cmap = 'bone_r')
# plt.title('Precip Frequency for Event '+str(int(i)))
# plt.colorbar()
# plt.show()
#%% Save dataset
Savingfolder = "/Users/kaying/Desktop/"
Events.to_netcdf(Savingfolder+str(P.time.values[0])[:10]+".nc")

    

