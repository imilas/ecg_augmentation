import os 
import re
import numpy as np
import pandas as pd
import ecg_plot
import neurokit2 as nk
from neurokit2_parallel import *
import torch.nn.functional as f
import torch.nn as nn
import math
import torch
from datetime import datetime
import importlib
from torch.utils.tensorboard import SummaryWriter
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from multiprocessing import Process
import multiprocessing
import gzip
import json
import pickle
import neurokit2 as nk
import matplotlib.pyplot as plt
import seaborn as sns

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
def label_shortner(l,trim_to=20):    
    label_shorts = {
        'Atrial_fibrillation':'Afib',
        'Supraventricular_tachycardia':'Supra. Tachycard',
        'Non-ST_elevation_(NSTEMI)_myocardial_infarction':'NSTEMI',
        'Pulmonary_embolism':'Pulmonary Emb.',
         'Syncope':'Syncope',
        'ST_elevation_(STEMI)_myocardial_infarction':'STEMI', 
        'Atrioventricular_block':'Atriovent. Block',
        'Hypertrophic_Cardiomyopathy':'Hypertroph. Cardio',
        'Aortic_Stenosis':'Aortic Stenosis', 
        'Cardiac_arrest':'Cardiac Arrest',
        'Mitral_Valve_Prolapse':'Mit.Valve Prolaps', 
        'Mitral_Valve_Stenosis':'Mit.Valve Sten.', 
        'Cardiac_Amyloidosis':'Cardiac Amyloid.',
        'Pulmonary_Hypertension':'Pulmon. Hypertens.', 
        'Heart_failure': 'Heart Failure', 
        'Unstable_angina': 'Unstable Angina',
        'Ventricular_tachycardia': 'Vent. Tachycardia'

    }
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
