# Exploring Best Practices for ECG Signal Processing in Machine Learning
## Requirements
This project makes heavy use of the tsai library (version 0.3.2). You can install

You would need to:
1. run this shell script to download physionet files: get_data.sh
2. run this notebook to convert the datasets to numpy arrays (this helps with very quick dataloading): parse_data.ipynb
3. run this notebook which shows how to make a dataloader and run one of TSAI's models on the data (you can use any pytorch compatible model) inception.ipynb