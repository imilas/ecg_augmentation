import os 
import re
import numpy as np
import pandas as pd
import ecg_plot
import math
from multiprocessing import Process
import multiprocessing
import gzip
import json
import pickle
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score,classification_report,precision_recall_fscore_support

def scale_sig(A):
    A -= A.min(1, keepdim=True)[0]
    A /= A.max(1, keepdim=True)[0] # consider that empty arrays will turn into Nans
    A = 2*A -1
    return A

def clean_sig(A,low_cut=2,high_cut=80):
    cleaned = ecg_clean(A.T).T
    cleaned = nk.signal_filter(cleaned, lowcut=low_cut, highcut=high_cut)
    return np.ascontiguousarray(cleaned)

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(path=""):
    with open(path, 'rb') as f:
        return pickle.load(f)



# shorten disease labels
def label_shortner(l,trim_to=20,label_shorts = False):
    if not label_shorts:
        label_shorts = {'sinus rhythm': 'Normal',
                         'atrial fibrillation': 'AF',
                         '1st degree av block': 'I-AVB',
                         'left bundle branch block': 'LBBB',
                         'right bundle branch block': 'RBBB',
                         'premature atrial contraction': 'PAC',
                         'ventricular ectopics': 'PVC',
                         'st depression': 'STD',
                         'st elevation': 'STE'}
    if l in list(label_shorts.keys()):
        return label_shorts[l]
    else:
        return l[0:trim_to]


def ecg_np_to_pandas(ecg,ecg_names=[]):
    # expects lead_number x lead_length shape (12x5500)
    if ecg_names:
        leads_names = ecg_names
    else:
        leads_names = 'I II III aVR aVL aVF V1 V2 V3 V4 V5 V6'.split(" ") 
    return pd.DataFrame(ecg,columns=leads_names)

def snomedConvert(label_df,snomed=True):
    codes =  pd.read_csv("data/snomed_codes.csv")
    if snomed:
        label_df.columns = [codes[codes["SNOMEDCTCode"] == int(x)]["Dx"].item() for x in label_df.columns]
        return label_df
def draw_aucs(y_test,y_prob,text_labels):
    n_classes = len(text_labels)
    y_test = y_test
    y_score = y_prob

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    font = {
            'weight' : 'bold',
            'size'   : 8}

    matplotlib.rc('font', **font)

    lw = 2 #line width
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this point
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(15,10))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = sns.color_palette('rainbow_r', 24)

    # sort keys by auc value 
    x = roc_auc.copy()
    del x["micro"]
    del x["macro"]

    sorted_classes = sorted(x, key=x.get,reverse=True)
    label_aucs = []
    for i, color in zip(sorted_classes, colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(text_labels[i], roc_auc[i]))
        label_aucs.append([text_labels[i], roc_auc[i]])


    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("title")
    plt.legend(loc="lower right")
#     plt.tight_layout()
    #     plt.savefig("images/%s.png"%title)
    #     plt.show()
    plt.ioff()
    return plt,label_aucs

def label_convert_ICBEB(label):
    mapping = {'1st degree av block':"I-AVB",
    'atrial fibrillation':"AF",
    'left bundle branch block':"LBBB",
    'premature atrial contraction':"PAC",
    'right bundle branch block':"RBBB",
    'sinus rhythm':"normal",
    'st depression':"STD",
    'st elevation':"STE",
    'ventricular ectopics':"PVC"}
    try: 
        return mapping[label]
    except:
        return label