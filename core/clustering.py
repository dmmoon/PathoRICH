import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {'0', '1', '2'}

import torch
from torchvision import models

import numpy as np


from tqdm import tqdm
from PIL import Image


import torch.nn as nn
import torchvision.transforms.functional as VF

from core.test import load_log

from collections import OrderedDict


import glob
from tqdm import tqdm

from openTSNE import TSNE

from sklearn.mixture import GaussianMixture

import shutil


import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

from config import N_CLUSTER, figsize



class ToTensor(object):
    def __call__(self, sample):
        img = VF.to_tensor(sample)
        return img


def load_ssl_network(pt_path):
    model = models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d).cuda()

    state_dict_weights = torch.load(pt_path)
    for i in range(4):
        state_dict_weights.popitem()
    state_dict_weights.keys()

    new_state_dict = OrderedDict()
    state_dict_init = model.state_dict()
    for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
        name = k_0
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.fc = torch.nn.Identity()
    return model


def execute_clustering(base, mag, model):
    transform = ToTensor()  
    mixed_vectors = []
    favorable_vectors, poor_vectors = [], []
    patch_labels = []
    with torch.no_grad():
        for dtype in ["favorable", "poor"]:
            fpaths = glob.glob(f"{base}/Attention Patches/{mag}/{dtype}/high score/*")
            if dtype == "favorable":
                label = 0
            else:
                label = 1
                
            for path in tqdm(fpaths):
                img = Image.open(path)
                img = transform(img)
                img = img.unsqueeze(dim=0).cuda()
                output = model(img).squeeze().detach().cpu().numpy().tolist()
                
                mixed_vectors.append(output)
                patch_labels.append(label)
                
                if dtype == "favorable":
                    favorable_vectors.append(output)
                else:
                    poor_vectors.append(output)

    
    for idx, vectors in enumerate([mixed_vectors, favorable_vectors, poor_vectors]):
        if not vectors:
            continue
        embedding = TSNE(random_state=41).fit(np.array(vectors))    
        km = GaussianMixture(n_components=N_CLUSTER, random_state=41)
        pca_labels = km.fit_predict(embedding)
        
        emb_pca = pd.DataFrame(embedding, columns=["x1", "x2"])
        emb_pca["cluster"] = pca_labels
        
        plt.figure(figsize=figsize)
        if idx == 0:
            emb_pca["label"] = patch_labels
            emb_pca["label"] = emb_pca["label"].apply(lambda x: "Poor" if x == 1 else "Favorable")
            sns.scatterplot(data=emb_pca, x="x1", y="x2", hue="cluster", palette=sns.color_palette("Spectral", as_cmap=True), style="label", legend='full').get_figure().savefig(f"Result/{os.path.basename(base)}/Attention Patches/{mag}/Mixed Case Clustering Map n {N_CLUSTER}.svg")
        else:
            if idx == 1:
                cls_name = "Favorable"
            else:
                cls_name = "Poor"
            sns.scatterplot(data=emb_pca, x="x1", y="x2", hue="cluster", palette=sns.color_palette("Spectral", as_cmap=True), legend='full').get_figure().savefig(f"Result/{os.path.basename(base)}/Attention Patches/{mag}/{cls_name} Case Clustering Map n {N_CLUSTER}.svg")
            

        favorable_patches = glob.glob(f"{base}/Attention Patches/{mag}/favorable/high score/*")
        poor_patches = glob.glob(f"{base}/Attention Patches/{mag}/poor/high score/*")

        if idx == 0:
            fpaths = favorable_patches + poor_patches
            cluster_base_path = os.path.join(base, "Attention Patches", mag, f"Mixed Patch Cluster")
            for i in range(N_CLUSTER):
                os.makedirs(os.path.join(cluster_base_path, str(i)), exist_ok=True)

            for fpath, cluster, label in zip(fpaths, pca_labels, patch_labels):
                if int(label) == 0:
                    label = "Favorable"
                else:
                    label = "Poor"
                shutil.copy(fpath, os.path.join(cluster_base_path, str(cluster), f"{label}_{os.path.basename(fpath)}"))
        else:
            if idx == 1:
                cluster_base_path = os.path.join(base, "Attention Patches", mag, f"Favorable Patch Cluster")
                fpaths = favorable_patches
            else:
                cluster_base_path = os.path.join(base, "Attention Patches", mag, f"Poor Patch Cluster")
                fpaths = poor_patches
                
            for i in range(N_CLUSTER):
                os.makedirs(os.path.join(cluster_base_path, str(i)), exist_ok=True)

            for fpath, cluster in zip(fpaths, pca_labels):
                shutil.copy(fpath, os.path.join(cluster_base_path, str(cluster), f"{os.path.basename(fpath)}"))            



def clustering_main():
    process_time = load_log()

    for mag in ["20x"]:
        pt_path = f"weights/simclr/model-{mag}.pth"
        model = load_ssl_network(pt_path)

        result_path = f"Result/{process_time}"
        execute_clustering(result_path, mag, model)
    
