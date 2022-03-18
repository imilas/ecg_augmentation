from helper_code import *
import numpy as np, os, sys, joblib
import ecg_plot
import pandas as pd
from glob import glob
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tsai.all import *
import torch
import optuna
from optuna.integration import FastAIPruningCallback
from sklearn.metrics import classification_report
import transformation_funcs as tfs
import seaborn as sns


torch.cuda.set_device(3) 


def snomedConvert(label_df,snomed=True):
    codes =  pd.read_csv("data/codes.csv",sep=";")[["Dx","SNOMED CT Code"]]
    if snomed:
        label_df.columns = [codes[codes["SNOMED CT Code"] == int(x)]["Dx"].item() for x in label_df.columns]
        return label_df
DATASET_NAME = "WFDB_CPSC2018"

X = np.load('./data/big_numpy_datasets/%s.npy'%DATASET_NAME, mmap_mode='c')
label_df = pd.read_csv("data/%s.csv"%DATASET_NAME).drop(columns=["headers","leads"])
y = snomedConvert(label_df)

# y=y[y.columns[y.sum()>(0.05*y.shape[0])]] # get rid of rare disease


splits = get_splits(y.to_numpy(), valid_size=.1,test_size=0.1, stratify=False, random_state=23, shuffle=True)


# df = pd.read_csv("data/%s.csv"%DATASET_NAME).drop(columns=["headers","leads"])
# y = snomedConvert(y)
y_multi = []
for i,row in y.iterrows():
    sample_labels = []
    for i,r in enumerate(row):
        if r == True:
            sample_labels.append(y.columns[i])
        
    y_multi.append(list(tuple(sample_labels)))
label_counts = collections.Counter([a for r in y_multi for a in r])
print('Counts by label:', dict(label_counts))

from transformation_funcs import *
def save_callback(study, trial):
    if study.best_trial == trial:
        PATH = Path('./models/inception_study_best.pkl')
        PATH.parent.mkdir(parents=True, exist_ok=True)
        global learn
        learn.export(PATH)
        
def objective(trial:optuna.Trial):    
    # Define search space here. More info here https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html
#     
    tfms = []
    random_shift = trial.suggest_float("random_shift", 0.0, 1, step=.1)
    noise = trial.suggest_float("noise", 0.0, 10, step=.5) 
    norm_type = trial.suggest_categorical('norm_type', ["Standardize", "Normalize"])
    CutOut = trial.suggest_categorical('cut_out', [True, False])
#     depth = trial.suggest_int('depth', 7,14,step=1) # search through all integer values between 3 and 9 with 3 increment steps
    scale = trial.suggest_float("scale", 0.05, 0.25, step=.05) 
    
    batch_tfms = [
        TSStandardize(by_sample=True),
        tfs.RandomShift(random_shift),
        tfs.MulNoise(noise),
        tfs.Scale(scale_factor=scale)
     ]
    
    if norm_type == "Normalize":
        batch_tfms[0] = tfs.Normalize
    if CutOut:
        batch_tfms.append(tfs.CutOutWhenTraining())
    
    tfms = [None,TSMultiLabelClassification()]

    dsets = TSDatasets(X.astype(float), y_multi, tfms=tfms, splits=splits) # inplace=True by default
    dls   = TSDataLoaders.from_dsets(dsets.train,dsets.valid, bs=[64, 128], batch_tfms=batch_tfms, num_workers=0)
    metrics = [precision_multi, recall_multi, specificity_multi, F1_multi] 

    model = InceptionTimePlus(dls.vars, dls.c, dls.len, depth=10,)
    global learn
    learn = Learner(dls, model, metrics=metrics,loss_func=nn.BCEWithLogitsLoss(), cbs=FastAIPruningCallback(trial,monitor="F1_multi"))
    learn.fit_one_cycle(30, lr_max=0.01)
    # get best f1 every scored
    f1 = np.max(np.array(learn.recorder.values)[:,-1])
    if f1>0.81:
        PATH = Path('./models/inception_best_%s.pkl'%f1)
        PATH.parent.mkdir(parents=True, exist_ok=True)
        learn.export(PATH)
    # Return the objective value
    return f1 # return the f1 value and try to maximize it

study_name = "inception_study" # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(study_name=study_name, storage=storage_name,direction='maximize',load_if_exists=True,
                            pruner=optuna.pruners.PatientPruner(optuna.pruners.NoPruner(),patience=25),
                           sampler=optuna.samplers.RandomSampler()
                           )

study.optimize(objective, n_trials=100,callbacks=[save_callback])

