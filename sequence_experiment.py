from helper_code import *
import numpy as np, os, sys, joblib
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
import argparse
import glob
import itertools 
import gc

parser = argparse.ArgumentParser(description='experiment parameters')
parser.add_argument('--max_len',default=8000,type=int,help="max_len of ecgs")
parser.add_argument('--gpu_num',default=0,type=int,help="gpu device")
parser.add_argument('--cv_num',default=0,type=int,help="cv num")
parser.add_argument('--arch',default="inception",help="inception or minirocket")
parser.add_argument('--dataset',default="CPSC2018",help="CPSC2018 or chapmanshaoxing or PTBXL")
# parser.add_argument('--cv_range',default=[0,1,2,3,4],nargs="+",type=int,help="folds to train")

args = parser.parse_args()
torch.cuda.set_device(args.gpu_num)
cv_num = args.cv_num
max_len = 4000
architecture = args.arch
DATASET_ID = args.dataset


# define the possible combinations
csv_path = "models/bandpassing/csvs/"
num_trials = 20
sc_values = [0.1, 0.25,0.5,0.75]
bp_values = [(1,30),(1,50),(1,100)]
norm_values = ["min_max"]

funcs = ["bp","sc","n"]
func_sequences = []
for i,f in enumerate(funcs):
    cfuncs = funcs.copy()
    cfuncs.pop(i)
    func_sequences.extend(list(zip([f]*len(cfuncs),cfuncs)))


# all possible csv names
experiment_names = []
for processing_type in func_sequences:
    bp = "bp" in processing_type
    n = "n" in processing_type
    sc = "sc" in processing_type
    
    if bp and sc:
        for pair in itertools.product(bp_values, sc_values):
            HP,LP = pair[0][0],pair[0][1]
            sf = pair[1]
            experiment_names.append("%s_%s_%s_%s_%s_%s_%s"%
                        (architecture,DATASET_ID,"-".join(processing_type),sf,HP,LP,"None"))
    if bp and n:
        for pair in itertools.product(bp_values, norm_values):
            HP,LP = pair[0][0],pair[0][1]
            norm_type = pair[1]
            experiment_names.append("%s_%s_%s_%s_%s_%s_%s"%(architecture,DATASET_ID,"-".join(processing_type),None,HP,LP,"minmax"))
    if sc and n:
        for pair in itertools.product(sc_values, norm_values):
            sf = pair[0]
            norm_type = pair[1]
            experiment_names.append("%s_%s_%s_%s_%s_%s_%s"%(architecture,DATASET_ID,"-".join(processing_type),sf,None,None,"minmax"))
         
DATASET_NAME = "WFDB_%s_signitured"%DATASET_ID
X = np.load('./data/big_numpy_datasets/%s.npy'%DATASET_NAME, mmap_mode='c')
label_df = pd.read_csv("data/%s.csv"%DATASET_NAME).drop(columns=["headers","leads"])
y = snomedConvert(label_df)
y = y[y.columns[y.sum()>0.05*len(y)] ]

cv_splits = get_splits(y.to_numpy(), n_splits = 20, valid_size=.1,test_size=0.1, stratify=False, random_state=23, shuffle=True)
y_multi = []
for i,row in y.iterrows():
    sample_labels = []
    for i,r in enumerate(row):
        if r == True:
            sample_labels.append(y.columns[i])
        
    y_multi.append(list(tuple(sample_labels)))
label_counts = collections.Counter([a for r in y_multi for a in r])

tfms  = [None, TSMultiLabelClassification()]

def float_if_valid(var):
    if var != "None":
        return np.float(var)
    else:
        return var
        
for experiment_name in experiment_names:
    ''' getting the function vars out of the experiment file name'''
    # experiment_name = experiment_names[0]
    vars = experiment_name.split("_")
    processing_type = vars[2].split("-")
    sf = float_if_valid(vars[3])
    HP,LP = float_if_valid(vars[4]),float_if_valid(vars[5])
    norm_type = vars[6]
    print("processing type: %s \n hp-lp:%s,%s \n scale factor:%s \n norm_type:%s"%
          (processing_type,HP,LP,sf,norm_type))
    batch_tfms = []
    for t in processing_type:
        if t == "sc":
            batch_tfms.append(tfs.Scale(scale_factor=sf,mode="nearest-exact"))
        if t == "n":
            batch_tfms.append(tfs.NormMinMax())
        if t == "bp":
            if sc == "None": # if there's no scaling, then we scale by 0.5 for faster processing
                batch_tfms.append(tfs.Scale(scale_factor=0.5,mode="nearest-exact"))
                batch_tfms.append(tfs.BandPass(int(0.5*500),low_cut=LP, high_cut=HP,leads=12,))
            else:
                 batch_tfms.append(tfs.BandPass(int(0.5*500),low_cut=LP, high_cut=HP,leads=12,))
    print("transforms:\n",[x.name for x in batch_tfms])
    csv_path = "models/sequences/csvs/%s_%s.csv"%(experiment_name,cv_num) ### CHANGE THIS 
    
    
    dsets = TSDatasets(X.astype(float)[:,:,0:max_len], y_multi, tfms=tfms, splits=cv_splits[cv_num]) # inplace=True by default
    dls   = TSDataLoaders.from_dsets(dsets.train,dsets.valid, bs=[64, 128], batch_tfms=batch_tfms, num_workers=0)
    metrics =[precision_multi, recall_multi, specificity_multi, F1_multi]
    if architecture == "inception":
        model = InceptionTimePlus(dls.vars, dls.c, dls.len,)
    elif architecture == "minirocket":
        model = MiniRocketPlus(dls.vars, dls.c,dls.len)
    elif architecture == "xresnet1d101":
        model = xresnet1d101(dls.vars, dls.c)
    # try : loss_func = BCEWithLogitsLossFlat(pos_weight=dls.train.cws.sqrt())
    
    learn = Learner(dls, model, metrics=metrics,
    #                     opt_func = wrap_optimizer(torch.optim.Adam,weight_decay=6.614e-07),
                    cbs=[fastai.callback.all.SaveModelCallback(
                        monitor="F1_multi",fname="%s_%s"%(experiment_name ,cv_num)),
                        fastai.callback.all.EarlyStoppingCallback(monitor='F1_multi', min_delta=0.005, patience=50)
                        ],
                    model_dir="models/sequences/")
    
    learn.fit_one_cycle(2, lr_max=0.01,)
    # now test it on test set
    learn.load("%s_%s"%(experiment_name ,cv_num))
    fold_splits = cv_splits[cv_num]
    dsets = TSDatasets(X.astype(float)[:,:,0:max_len], y_multi, tfms=tfms, splits=(fold_splits[0],fold_splits[2])) # inplace=True by default
    dls   = TSDataLoaders.from_dsets(dsets.train,dsets.valid, bs=[128, 128], batch_tfms=batch_tfms, num_workers=0)
    
    valid_probas, valid_targets, valid_preds = learn.get_preds(dl=dls.valid, with_decoded=True)
    y_pred = (valid_preds>0)
    y_test = valid_targets
    report = classification_report(y_test, y_pred,target_names = dls.vocab.o2i.keys(),digits=3,output_dict=True)
    df = pd.DataFrame(report).reset_index()
    df.to_csv(csv_path,index=False)
    del model,learn
    gc.collect()
    torch.cuda.empty_cache()

    # break ### remove break 

# processing_type