# Exploring Best Practices for ECG Signal Processing in Machine Learning

## What this project does:
In this project we:
1. Download and process all physionet ECG files for quick dataloading. 
2. Deploy pytorch models on these datasets (courtesy of TSAI:https://github.com/timeseriesAI/tsai) 
    - Any pytorch model can be used here, dataloaders can also be modified. 
3. Create custom signal processing functions to transform datasets before they are given to the model.
    - You add your own in transformation_funcs.py

## Requirements
This project makes heavy use of the tsai library (version 0.3.2), which requires pytorch. We recommend use of the conda environment. 
You can clone our conda environment with ```conda create --name <env> --file requirements.txt```
You can also choose to install the libraries via pip. 

## How to get started:
1. run this shell script to download physionet files: get_data.sh
2. run parse_data.ipynb notebook to convert the datasets to numpy arrays (this helps with very quick dataloading)
3. run inception.ipynb to see how to make a dataloaders and run one of TSAI's models on the data (you can use any pytorch compatible model): 
4. run transformation.ipynb to play around with transformations
5. run experiment_analysis_*.ipynb to see how we analyzed experiment results