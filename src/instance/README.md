
## Description
Instance parameters we have:

1. `instance_id` - The ID of the instance.
2. `N` - The number of scenarios to be read for each sample.
3. `M` - The number of samples to be read for testing solution quality.
4. `T` - The number of time periods.
5. `Q` - The number of capacity levels considered in each satellites
6. `alpha` - The weight of the cost of serving from satellites respect to the cost of serving from DCs.
   1. Review parameter C from ARCE functions.
   2. if `alpha > 1` then the cost of serving from satellites is cheaper than serving from DCs.
7. `beta` - The weight of the cost of installing a satellite respect to the cost of operating a satellite.
   1. if `beta > 1` then the cost of installing a satellite is more expensive than the cost of operating a satellite.
8. `type_of_flexibility` - The type of flexibility to be considered.
   1. `type_of_flexibility = 1` - The flexibility is considered in the installation of satellites, i.e., if satellites are installed so that they have to operate always with the same capacity.
   2. `type_of_flexibility = 2` - The flexibility is considered in the installation and operation of satellites, i.e., if satellites are installed so that they can operate with different capacities.
9. `periods` - The periods to be considered.
10. `satellites` - The satellites to be installed.
    1. `cost installation` - The cost of installing the satellite.
    2. `cost operation` - The cost of operating the satellite.
11. `scenarios` - The scenarios obtained from sampling the demand.
    1.  `pixel` - The pixel of the scenario.
    2.  `costs` - The costs of serving from DCs and satellites.
    3.  `fleet size required` - The fleet size required to serve the demand.
