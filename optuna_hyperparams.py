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


torch.cuda.set_device(4) 


def snomedConvert(label_df,snomed=True):
    codes =  pd.read_csv("data/codes.csv",sep=";")[["Dx","SNOMED CT Code"]]
    if snomed:
        label_df.columns = [codes[codes["SNOMED CT Code"] == int(x)]["Dx"].item() for x in label_df.columns]
        return label_df
DATASET_NAME = "WFDB_CPSC2018"


# In[2]:


X = np.load('./data/big_numpy_datasets/%s_signitured.npy'%DATASET_NAME, mmap_mode='c')
label_df = pd.read_csv("data/%s.csv"%DATASET_NAME).drop(columns=["headers","leads"])
y = snomedConvert(label_df)
# get diseases that exist in more than 0.5 percent of the samples
y=y[y.columns[y.sum()>(0.005*y.shape[0])]]
y.sum()
y.reset_index(drop=True)


# In[3]:


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


# In[ ]:


def objective(trial:optuna.Trial):    
    # Define search space here. More info here https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html
#     
    tfms = []
    depth = trial.suggest_int('depth', 8,13,step=1) # search through all integer values between 3 and 9 with 3 increment steps
    kernel_size = trial.suggest_int('kernel_size', 100,200,step=10) # search through all integer values between 3 and 9 with 3 increment steps
    scale = trial.suggest_uniform('scale', 0.2,0.7) 
    max_length = trial.suggest_int("max_length",3000,8000,step = 1000)
#     use_loss_weights = trial.suggest_categorical("loss_weights", [True,False]) 
    batch_tfms = [
        tfs.Normalize(),
        tfs.Scale(scale_factor=scale)
     ]

    
    tfms = [None,TSMultiLabelClassification()]
    splits = get_splits(y.to_numpy(), valid_size=0.25, stratify=False, random_state=23, shuffle=True)
    dsets = TSDatasets(X.astype(float)[:,:,0:max_length], y_multi, tfms=tfms, splits=splits) # inplace=True by default
    dls   = TSDataLoaders.from_dsets(dsets.train,dsets.valid, bs=[64, 128], batch_tfms=batch_tfms, num_workers=0)
    metrics = [precision_multi, recall_multi, specificity_multi, F1_multi] 
    model = InceptionTimePlus(dls.vars, dls.c, dls.len, depth=depth,ks=kernel_size)
    
    learn = Learner(dls, model, metrics=metrics,loss_func=nn.BCEWithLogitsLoss(),
                    opt_func = wrap_optimizer(torch.optim.Adam,),
                    cbs=FastAIPruningCallback(trial,monitor="F1_multi"))
    learn.recorder.silent = True 
    with learn.no_logging():
        with learn.no_bar():
            learn.fit_one_cycle(300, lr_max=0.01)
    # get best f1 every scored
    f1 = np.max(np.array(learn.recorder.values)[:,-1])
    # save model if u want
#     if f1>0.84:
#         PATH = Path('./models/inception_hyperparam_%s.pkl'%f1)
#         PATH.parent.mkdir(parents=True, exist_ok=True)
#         learn.export(PATH)
    # Return the objective value
    return f1 # return the f1 value and try to maximize it

study_name = "hyperparam_search_%s"%DATASET_NAME # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(study_name=study_name, storage=storage_name,direction='maximize',load_if_exists=True,
                            pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(),patience=50),
                           )

optuna.logging.set_verbosity(optuna.logging.CRITICAL)

study.optimize(objective, n_trials=50,)

