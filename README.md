# Exploring Best Practices for ECG Signal Processing in Machine Learning
## What this project does:
In this project we:
1. Download and process all physionet ECG files for quick dataloading. 
2. Deploy pytorch models on these datasets
3. Create c
## Requirements
This project makes heavy use of the tsai library (version 0.3.2), which requires pytorch. We recommend use of the conda environment. 
You can clone our conda environment with ```conda create --name <env> --file requirements.txt```
You can also choose to install the libraries via pip. 

## How to get started:
1. run this shell script to download physionet files: get_data.sh
2. run this notebook to convert the datasets to numpy arrays (this helps with very quick dataloading): parse_data.ipynb
3. run this notebook which shows how to make a dataloader and run one of TSAI's models on the data (you can use any pytorch compatible model) inception.ipynb