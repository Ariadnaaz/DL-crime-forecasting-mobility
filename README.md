# DL-crime-forecasting-mobility

This repository contains all the code used for the paper "Deep Learning for Crime Forecasting: The Role of Mobility at Fine-grained Spatiotemporal Scales" by Ariadna Albors Zumel, Michele Tizzoni, and Gian Maria Campedelli.

The code is structured as follows:

## Preprocessing
- Sociodemographic data:
  - Raw data: Folder containing the raw data to be prepossessed. This data was obtained from the 5-years estimates of ASC.
  - `1.Preprocess_ data_sociodem_datasets.ipynb`: First file to start the preprocessing of this data.
  - `2.Make_sociodem_grids.ipynb`: Second and last file to finish the preprocessing of this data.
 
- Crime data:
  - Raw data: Folder containing the raw data to be prepossessed. This data was obtained from the...
  - `1.Generate_selected_crimes_datasets.ipynb`:
  - `2.Assign_grid_cell_crime_data.ipynb`:
  - `3.Make_crime_final_files_with_grid_cells.ipynb`:
 
- Mobility data: (doesn't contain a raw data folder since this dataset was accessed through a payed subscription)
  - `1.Downloading_mobility_dataset_Advan.ipynb`:
  - `2.Preprocess_mobility_data.ipynb`:
  - `3.Assign_grid_cell_mobility_data.ipynb`:
  

## Forecasting models
