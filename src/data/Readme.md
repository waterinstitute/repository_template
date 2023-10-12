# Documentation for the Data Processing in Azure Batch

Feature engineering is done by processing digital elevation models (DEM) and land cover/use maps (from the USDA) into feature indicative of flooding

## Data sources

### DEM

DEMs are downloaded from the USGS based on a call to the following API to get the file location on Amazon s3:
https://apps.nationalmap.gov/tnmaccess/#/product

The files also may be manually searched by accessing teh following link:

https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/Elevation/

### NLCD land cover

National Land Cover Database: https://www.usgs.gov/centers/eros/science/national-land-cover-database

## Features

As a precusor to building a feature store for Insight, we list the features developed in the preprocessing stage of Azure Batch and indicate which are used for flash and riverine flooding:

1. Topographic Wetness Index (twi): logarithm of the ration between flow accumulation area and tangent of the slope for each cell
    - type: float
    - numerical range: real. Values below -1 are uncommon. Data validation to issue warning for values below -1 and above 40.
    - references: 
        - https://grass.osgeo.org/grass76/manuals/r.watershed.html
        - https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/how-flow-accumulation-works.htm
        - https://en.wikipedia.org/wiki/Topographic_wetness_index
2. geofloodindex (local and nonlocal): combining finalelevationstream, distancestream, finalcontribarea, and the Mannings velocity for open-channel flows
    - type: float
    - numerical range: real. Values below -2 are uncommon. Data validation to issue warning for values below -1 and above 20.
    - references:
        - https://en.wikipedia.org/wiki/Manning_formula
        - creator: Mark S. Bartlett
3. finalelevationstream (local and nonlocal): elevation above stream along watercourse
    - type: float
    - numerical range: positive value. Throw warning if any data is negative and exception if more than 1% of data are negative and warning if any value is above 250.
    - references:
        - https://grass.osgeo.org/grass78/manuals/addons/r.stream.distance.html
4. distancestream (local and nonlocal): distance above stream along watercourse
    - type: float
    - numerical range: positive value. Throw exception if any value is negative and warning if any value is above 10,000.
    - references:
        - https://grass.osgeo.org/grass78/manuals/addons/r.stream.distance.html
5. finalcontribarea (local and nonlocal): Difference between the accumulation at a given cell and the nearest stream's contributing area
    - type: float
    - numerical range: positive value. Throw exception if any value is negative and warning if any value is above 20,000,000.
6. riverinefloodindex_f_drain_flow_100 (local and nonlocal):
    - type:
    - numerical range:
7. riverinefloodindex_f_drain_flow_100 (local and nonlocal):
    - type:
    - numerical range:

Features 2 through 7 have both a local and nonlocal property.

Features 1 through 5 area used for flash flood prediction while feautres 1, 6, and 7 are used for rivering flood prediction.

## Data validation

The data validation script `data_validation.py` is run inside Databricks to ensure the features:
1. have the right data type
2. do not contain null values after removing rows corresponding to buffer areas
3. do not contain an abnormally high number of zeros
4. has flash and riverine heatmap values above zero (for training data)
5. have values within acceptable bounds

Tests 1 and 2 raise exceptions when violated while tests 3, 4, and 5 issue warnings.

## Writing the Delta Table

The final dataframe is saved as a Delta Table https://docs.delta.io/latest/delta-batch.html#write-to-a-table&language-python. The final dataframe compresses each row to contain a tile of 256x256 cells or less (if at the boundary of the watershed/buffer). The dataframe needs to be exploded before working with it.

