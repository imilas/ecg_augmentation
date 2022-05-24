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
# summary(sapply(c(1:1000), function(i) { wilcox.test(rnorm(20),rnorm(20)+1)$p.value } ))
# 
# > f <- function(n) { summary(sapply(c(1:1000), function(i) { wilcox.test(0.85 + 0.1*runif(n),0.85 + 0.1 * 1.1 * runif(n))$p.value } )) }
# > f(5)
#     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# 0.007937 0.287698 0.547619 0.540103 0.841270 1.000000 
# > f(10)
#     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# 0.002089 0.190316 0.481251 0.485247 0.739364 1.000000 
# > f(15)
#     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# 0.000394 0.228823 0.460959 0.487258 0.743702 1.000000 
# > f(20)
  
# > f(40)
#      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
# 0.0000362 0.1565877 0.3806263 0.4177091 0.6631989 1.0000000 
# > f(100)
#     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# 0.000005 0.069457 0.241847 0.330647 0.563358 0.999025 
# > f(1000)
#      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
# 0.0000000 0.0000346 0.0004677 0.0139595 0.0048003 0.6399986 
# > f(40)
#      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
# 0.0000362 0.1565877 0.3806263 0.4177091 0.6631989 1.0000000 
# > f(100)
#     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# 0.000005 0.069457 0.241847 0.330647 0.563358 0.999025 
# > f(1000)
#      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
# 0.0000000 0.0000346 0.0004677 0.0139595 0.0048003 0.6399986 
# > g <- function(n) { summary(sapply(c(1:1000), function(i) { wilcox.test(0.85 + 0.1*runif(n),0.85 + 0.15 * runif(n))$p.value } )) }
# > g(5)
#     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# 0.007937 0.150794 0.309524 0.420468 0.690476 1.000000 
# > g(10)
#      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
# 0.0000433 0.0524259 0.1903159 0.3034662 0.4812509 1.0000000 
# > g(20)
#    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.00000 0.01430 0.07627 0.17925 0.24227 1.00000
# > g(40)
#      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
# 0.0000003 0.0009403 0.0082350 0.0547694 0.0476009 0.9580311 
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
batch_tfms = args.batch_tfms

processing_type = '-'.join([x for x in args.batch_tfms])

if "sc" in args.batch_tfms:
    batch_tfms.append(tfs.Scale(scale_factor=sf,mode=scale_type))

if "n" in args.batch_tfms:
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
        
if "bp" in args.batch_tfms:
    batch_tfms.append(tfs.BandPass(int(sf*500),low_cut=50, high_cut=1,leads=12,))
if "sh" in args.batch_tfms:
    batch_tfms.append(tfs.RandomShift(0.1))
if len(args.batch_tfms)==0:
    processing_type = "raw"
print("transforms:",[x.name for x in batch_tfms])
print(processing_type)


def snomedConvert(label_df,snomed=True):
    codes =  pd.read_csv("data/snomed_codes.csv",sep=",")[["Dx","SNOMEDCTCode"]]
    if snomed:
        label_df.columns = [codes[codes["SNOMEDCTCode"] == int(x)]["Dx"].item() for x in label_df.columns]
        return label_df

DATASET_NAME = "WFDB_%s_signitured"%DATASET_ID
X = np.load('./data/big_numpy_datasets/%s.npy'%DATASET_NAME, mmap_mode='c')
label_df = pd.read_csv("data/%s.csv"%DATASET_NAME).drop(columns=["headers","leads"])
y = snomedConvert(label_df)


cv_splits = get_splits(y.to_numpy(), n_splits = 5, valid_size=.1,test_size=0.1, stratify=False, random_state=23, shuffle=True)
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
                    cbs=[fastai.callback.all.SaveModelCallback(monitor="F1_multi",fname="%s_%s_%s_%s_%s_%s_%s"%(architecture,DATASET_ID,processing_type,sf,scale_type,norm_type,cv_num))],
                    model_dir="models/5CV/")
    with learn.no_logging():
        with learn.no_bar():
            learn.fit_one_cycle(300, lr_max=0.01)



