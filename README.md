# DL-crime-forecasting-mobility

This repository contains all the code used for the paper "Deep Learning for Crime Forecasting: The Role of Mobility at Fine-grained Spatiotemporal Scales" by Ariadna Albors Zumel, Michele Tizzoni, and Gian Maria Campedelli.

The code is separated into two main folders and structured as follows.

## Preprocessing
- Sociodemographic data:
  - Raw data: Folder containing the raw data to be prepossessed. This data was obtained from the 5-years estimates of ASC.
  - `1-preprocess_ data_sociodem_datasets.py`: First file to start the preprocessing of this data.
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
  

## Forecasting models
