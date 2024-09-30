import os
import shutil
import dgl.random
import torch
import numpy as np
import random
import ruamel.yaml as yaml
import argparse
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    dgl.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_directories(root, calibrator, ds_name):
    if os.path.exists(os.path.join(root, calibrator, ds_name)):
        shutil.rmtree(os.path.join(root, calibrator, ds_name))
    os.makedirs(os.path.join(root, calibrator, ds_name))
    os.makedirs(os.path.join(root, calibrator, ds_name, "confidence"))
    os.makedirs(os.path.join(root, calibrator, ds_name, "accuracy"))
    os.makedirs(os.path.join(root, calibrator, ds_name, "diff"))
    # os.makedirs(os.path.join(root, calibrator, ds_name, "diff_diff"))


def load_conf(path:str = None, dataset:str = None):
    if path == None:
        dir = "config"
        path = os.path.join(dir, dataset+".yaml")
        if os.path.exists(path) == False:
            raise KeyError("The configuration file is not provided.")
    
    conf = open(path, "r").read()
    conf = yaml.load(conf)

    if conf['calibration']['calibrator_name'] == 'GETS':
        conf = open("gets_"+path, "r").read()
        conf = yaml.load(conf)

    import nni
    if nni.get_trial_id()!="STANDALONE":
        par = nni.get_next_parameter()
        for i, dic in conf.items():
            if type(dic) == type(dict()):
                for a,b in dic.items():
                    for x,y in par.items():
                        if x == a:
                            conf[i][a] = y
            for x,y in par.items():
                if x == i:
                    conf[i] = y
                    
    conf = argparse.Namespace(**conf)

    return conf

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)