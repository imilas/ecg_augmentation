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
parser.add_argument('--max_len',default=8000,type=int,help="max_len of ecgs")
parser.add_argument('--scale',type=float,default=0.5,help="down/upsample scale")
parser.add_argument('--gpu_num',default=0,type=int,help="gpu device")
parser.add_argument('--arch',default="inception",help="inception or minirocket")
parser.add_argument('--dataset',default="CPSC2018",help="CPSC2018 or chapmanshaoxing")
parser.add_argument('--cv_range',default=[0,1,2,3,4,5,6,7,8,9],nargs="+",type=int,help="folds to train")

args = parser.parse_args()

print("pre-processing funcs: ",args.batch_tfms)
print("gpu num :",args.gpu_num)
print("scaling :",args.scale)
print("training folds:",args.cv_range)
print("max len:",args.max_len)
print("dataset",args.datset)

torch.cuda.set_device(args.gpu_num)
max_len = args.max_len
sf = args.scale
cv_range = args.cv_range
architecture = args.arch
DATASET_ID = args.dataset
batch_tfms = []

# batch_tfms = [
#                 tfs.Scale(scale_factor=sf,),
#                 tfs.Normalize(),
#                 tfs.BandPass(int(sf*500),low_cut=50, high_cut=1,leads=12,),
#                 tfs.RandomShift(0.1),
#                 tfs.MulNoise(6),
#                 tfs.CutOutWhenTraining(),
#              ]

processing_type = '-'.join([x for x in args.batch_tfms])
if "sc" in args.batch_tfms:
    batch_tfms.append(tfs.Scale(scale_factor=sf,))
if "n" in args.batch_tfms:
    batch_tfms.append(tfs.Normalize())
if "bp" in args.batch_tfms:
    batch_tfms.append(tfs.BandPass(int(sf*500),low_cut=50, high_cut=1,leads=12,))
if "sh" in args.batch_tfms:
    batch_tfms.append(tfs.RandomShift(0.1))
if len(args.batch_tfms)==0:
    processing_type = "raw"
print("transforms:",[x.name for x in batch_tfms])
print(processing_type)

# torch.cuda.set_device(3)
# cv_range=[0,1,2,3]

# torch.cuda.set_device(2)
# cv_range=[4,5,6]

# torch.cuda.set_device(4)
# cv_range=[7,8,9]


def snomedConvert(label_df,snomed=True):
    codes =  pd.read_csv("data/snomed_codes.csv",sep=",")[["Dx","SNOMEDCTCode"]]
    if snomed:
        label_df.columns = [codes[codes["SNOMEDCTCode"] == int(x)]["Dx"].item() for x in label_df.columns]
        return label_df

DATASET_NAME = "WFDB_%s_signitured"%DATASET_ID
X = np.load('./data/big_numpy_datasets/%s.npy'%DATASET_NAME, mmap_mode='c')
label_df = pd.read_csv("data/%s.csv"%DATASET_NAME).drop(columns=["headers","leads"])
y = snomedConvert(label_df)


cv_splits = get_splits(y.to_numpy(), n_splits = 10, valid_size=.1,test_size=0.1, stratify=False, random_state=23, shuffle=True)
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


for cv_num in cv_range:
#     cv_num = cv_num + 0
    dsets = TSDatasets(X.astype(float)[:,:,0:max_len], y_multi, tfms=tfms, splits=cv_splits[cv_num]) # inplace=True by default
    dls   = TSDataLoaders.from_dsets(dsets.train,dsets.valid, bs=[64, 128], batch_tfms=batch_tfms, num_workers=0)
    metrics =[accuracy_multi, balanced_accuracy_multi, precision_multi, recall_multi, specificity_multi, F1_multi]
    if architecture == "inception":
        model = InceptionTimePlus(dls.vars, dls.c, dls.len, depth=12, ks = 130,nf=32 )
    elif architecture == "minirocket":
        model = MiniRocketPlus(dls.vars, dls.c,dls.len)
    # try : loss_func = BCEWithLogitsLossFlat(pos_weight=dls.train.cws.sqrt())
    
    learn = Learner(dls, model, metrics=metrics,
#                     opt_func = wrap_optimizer(torch.optim.Adam,weight_decay=6.614e-07),
                    cbs=[fastai.callback.all.SaveModelCallback(monitor="F1_multi",fname="%s_%s_%s_%s_%s"%(architecture,DATASET_ID,processing_type,max_len,cv_num))],
                    model_dir="models/10CV/")
    with learn.no_logging():
        with learn.no_bar():
            learn.fit_one_cycle(300, lr_max=0.01)



