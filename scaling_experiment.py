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
from torchsummary import summary
import argparse

parser = argparse.ArgumentParser(description='experiment parameters')
parser.add_argument('--batch_tfms',nargs="+",default=[],help="input list of ints ->n:normalize sc:scale bp:bandpass sh:shift")
parser.add_argument('--norm_type',default="minmax",help="normalization function (minmax, maxdiv,zscore,median,deci_scale)")
parser.add_argument('--max_len',default=8000,type=int,help="max_len of ecgs")
parser.add_argument('--scale',type=float,default=0.5,help="down/upsample scale")
parser.add_argument('--scale_type',default="nearest-exact",help="nearest / nearest-exact / area")
parser.add_argument('--gpu_num',default=0,type=int,help="gpu device")
parser.add_argument('--arch',default="inception",help="inception or minirocket")
parser.add_argument('--dataset',default="CPSC2018",help="CPSC2018 or chapmanshaoxing")
parser.add_argument('--cv_range',default=[0,1,2,3,4],nargs="+",type=int,help="folds to train")

args = parser.parse_args()

print("pre-processing funcs: ",args.batch_tfms)
print("gpu num :",args.gpu_num)
print("scaling :",args.scale)
print("training folds:",args.cv_range)
print("max len:",args.max_len)
print("dataset",args.dataset)

torch.cuda.set_device(args.gpu_num)
norm_type = args.norm_type
max_len = args.max_len
sf = args.scale
scale_type = args.scale_type
cv_range = args.cv_range
architecture = args.arch
DATASET_ID = args.dataset
transforms = args.batch_tfms
batch_tfms = []
processing_type = '-'.join([x for x in transforms])

if "sc" in transforms:
    batch_tfms.append(tfs.Scale(scale_factor=sf,mode=scale_type))

if "n" in transforms:
    if norm_type == "minmax":
        batch_tfms.append(tfs.NormMinMax())
    if norm_type == "maxdiv":
        batch_tfms.append(tfs.NormMaxDiv())
    if norm_type == "zscore":
        batch_tfms.append(tfs.NormZScore())
    if norm_type == "median":
        batch_tfms.append(tfs.NormMedian())
    if norm_type == "deci_scale":
        batch_tfms.append(tfs.NormDecimalScaling())
        
if "bp" in transforms:
    batch_tfms.append(tfs.BandPass(int(sf*500),low_cut=50, high_cut=1,leads=12,))
if "sh" in transforms:
    batch_tfms.append(tfs.RandomShift(0.1))
if len(transforms)==0:
    processing_type = "raw"
print("transforms:",[x.name for x in batch_tfms])
print(processing_type)


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
print('Counts by label:', dict(label_counts))
tfms  = [None, TSMultiLabelClassification()]

for cv_num in range(20):
#     cv_num = cv_num + 0
    dsets = TSDatasets(X.astype(float)[:,:,0:max_len], y_multi, tfms=tfms, splits=cv_splits[cv_num]) # inplace=True by default
    dls   = TSDataLoaders.from_dsets(dsets.train,dsets.valid, bs=[64, 128], batch_tfms=batch_tfms, num_workers=0)
    metrics =[precision_multi, recall_multi, specificity_multi, F1_multi]
    if architecture == "inception":
        model = InceptionTimePlus(dls.vars, dls.c, dls.len,)
    elif architecture == "minirocket":
        model = MiniRocketPlus(dls.vars, dls.c,dls.len)
    # try : loss_func = BCEWithLogitsLossFlat(pos_weight=dls.train.cws.sqrt())
    
    learn = Learner(dls, model, metrics=metrics,
#                     opt_func = wrap_optimizer(torch.optim.Adam,weight_decay=6.614e-07),
                    cbs=[fastai.callback.all.SaveModelCallback(
                        monitor="F1_multi",fname="%s_%s_%s_%s_%s"%(architecture,DATASET_ID,processing_type,sf,cv_num)),
                        fastai.callback.all.EarlyStoppingCallback(monitor='F1_multi', min_delta=0.005, patience=50)
                        ],
                    model_dir="models/scaling/")

    learn.fit_one_cycle(300, lr_max=0.01,)
    # now test it on test set
    learn.load("%s_%s_%s_%s_%s"%(architecture,DATASET_ID,processing_type,sf,cv_num))
    fold_splits = cv_splits[cv_num]
    dsets = TSDatasets(X.astype(float)[:,:,0:max_len], y_multi, tfms=tfms, splits=(fold_splits[0],fold_splits[2])) # inplace=True by default
    dls   = TSDataLoaders.from_dsets(dsets.train,dsets.valid, bs=[128, 128], batch_tfms=batch_tfms, num_workers=0)

    valid_probas, valid_targets, valid_preds = learn.get_preds(dl=dls.valid, with_decoded=True)
    y_pred = (valid_preds>0)
    y_test = valid_targets
    report = classification_report(y_test, y_pred,target_names = dls.vocab.o2i.keys(),digits=3,output_dict=True)
    df = pd.DataFrame(report).reset_index()
    df.to_csv("models/scaling/csvs/%s_%s_%s_%s_%s.csv"%(architecture,DATASET_ID,processing_type,sf,cv_num),index=False)
    df


