## Description

This is a collection of data sets that I have used in my projects. The data sets are in the `data` folder. The `data` folder also contains a `README.md` file for all the data sets. The `README.md` file contains the description of the data sets and the source of the data sets.

## Data Sets

### Satellites
This data set contains the information about the satellites that are usign in second-echelon. The data set contains the following information about the satellites:
- id_satellite
- location (longitude, latitude)
- distance from DC (in km)
- travel_time from DC (in hours)
- travel_time from DC in traffic (in hours)
- capacity (in # of parking spots)
- cost fixed (in $usd) depends on the capacity.
- cost operation (in $usd) depends on the capacity and period of time.
- cost sourcing (in $usd/item).

### Pixels
This data set contains the information about the pixels that represent clusters of customer . The data set contains the following information about the pixels:
- id_pixel
- location (longitude, latitude)
- area surface (in km^2)
- speed intra stop (in km/h).
- constant K.

### Instances - Scenario $i^{\text{th}}$
This data set contains the information about the instances that are used in the project. The data set contains the following information about the instances:

- id_instance
- id_pixel
- demand by period of time (in # of items per period of time).
- avg drop size by period of time (in drop per period of time).
- avg stop by period of time (in stop per period of time).


### Distance Matrix DC to Pixels
This data set contains the information about the distance matrix between the DC and the pixels. The data set contains the following information about the distance matrix:

- id_pixel
- distance from DC (in km)
- travel_time from DC (in hours)
- travel_time from DC in traffic (in hours)

### Distance Matrix Satellites to Pixels
This data set contains the information about the distance matrix between the satellites and the pixels. The data set contains the following information about the distance matrix:

- id_satellite
- id_pixel
- distance from satellite (in km)
- travel_time from satellite (in hours)
- travel_time from satellite in traffic (in hours)
