# DL-crime-forecasting-mobility

This repository contains all the code used for the paper "Deep Learning for Crime Forecasting: The Role of Mobility at Fine-grained Spatiotemporal Scales" by Ariadna Albors Zumel, Michele Tizzoni, and Gian Maria Campedelli.

📄 You can find the full paper here:

## Introduction

**Objectives:** To develop a deep learning framework to evaluate if and how incorporating micro-level mobility features, alongside historical crime and sociodemographic data, enhances predictive performance in crime forecasting at fine-grained spatial and temporal resolutions.

**Methods:** We advance the literature on computational methods and crime forecasting by focusing on four U.S.\ cities (i.e., Baltimore, Chicago, Los Angeles, and Philadelphia). We employ crime incident data obtained from each city's police department, combined with sociodemographic data from the American Community Survey and human mobility data from Advan, collected from 2019 to 2023. This data is aggregated into grids with equally sized cells of 0.077 sq. miles (0.2 sq. kms) and used to train our deep learning forecasting model that predicts crime occurrences 12 hours ahead using 14-day and 2-day input sequences.

**Results:** Incorporating mobility features improves predictive performance, especially when using shorter input sequences. Noteworthy, however, the best results are obtained when both mobility and sociodemographic features are used together, with our deep learning model achieving the highest recall, precision, and F1 score in all four cities, outperforming alternative methods. With this configuration, longer input sequences enhance predictions for violent crimes, while shorter sequences are more effective for property crimes.

**Conclusion:** These findings underscore the importance of integrating diverse data sources for spatiotemporal crime forecasting, mobility included. They also highlight the advantages (and limits) of deep learning when dealing with fine-grained spatial and temporal scales.


## Code structure

The code is separated into two main folders and structured as follows.

### Preprocessing
- Sociodemographic data:
  - Download the data from [ACS](https://www2.census.gov/geo/tiger/TIGER_DP/) (specifically the files containing BG_06, BG_17, BG_24, and GB_42 for the year 2019, 2020 and 2021) and put in a folder named "raw_data".
  - `1-preprocess_data_sociodem_datasets.py`: First file to start the preprocessing of this data.
  - `2-make_sociodem_grids.py`: Second and last file to finish the preprocessing of this data.
 
- Crime data:
  - Raw data: Folder containing the raw data to be prepossessed. This data was obtained from the...
  - `1-generate_selected_crimes_datasets.py`:
  - `2-assign_grid_cell_crime_data.py`:
  - `3-make_crime_final_files_with_grid_cells.py`:
 
- Mobility data: (doesn't contain a raw data folder since this dataset was accessed through a payed subscription)
  - `1-downloading_mobility_dataset_Advan.py`:
  - `2-preprocess_mobility_data.py`:
  - `3-assign_grid_cell_mobility_data.py`:

### Forecasting models


## Model architecture
