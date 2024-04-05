import core.dsmil as mil

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, glob, copy
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
from collections import OrderedDict
from sklearn.utils import shuffle

from config import LOWSCALE_CANCER_THRESHOLD, HIGHSCALE_CANCER_THRESHOLD

import random
from torch.backends import cudnn

torch.manual_seed(32)
torch.cuda.manual_seed(32)
np.random.seed(32)
random.seed(32)
cudnn.deterministic = True
cudnn.benchmark = False


class BagDataset:
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        img = os.path.join(temp_path)
        img = Image.open(img)

        # sample = {'input': img}

        if self.transform:
            img = self.transform(img)
        return img


# class ToTensor(object):
#     def __call__(self, sample):
#         img = sample['input']
#         img = VF.to_tensor(img)
#         return {'input': img}


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=Compose([
                                        transforms.ToTensor(),
                                    ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)


def get_tumor_patch_list(tumor_patch_txt):
    mag_lows = set()
    mag_highs = set()
    
    if tumor_patch_txt is None:
        return None, None
    
    list_5x_txt, list_20x_txt = tumor_patch_txt
    
    for idx, txt_path in enumerate([list_5x_txt, list_20x_txt]):
        with open(txt_path, "r") as f:
            for line in f.readlines():
                path, cancer_ratio = line.strip().split(",")
                
                if idx == 0:
                    CANCER_THRESHOLD = LOWSCALE_CANCER_THRESHOLD
                else:
                    CANCER_THRESHOLD = HIGHSCALE_CANCER_THRESHOLD
                
                if float(cancer_ratio) >= CANCER_THRESHOLD:
                    if idx == 0:
                        mag_lows.add(os.path.normpath(path))
                    else:
                        mag_highs.add(os.path.normpath(path))
    return mag_lows, mag_highs




def compute_feats(args, bags_list, i_classifier, save_path=None, magnification='single', tumor_patch_txt=None, xlsx_path=None):
    i_classifier.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    
    candidate_table_5x, candidate_table_20x = get_tumor_patch_list(tumor_patch_txt)
    
    for i in range(0, num_bags):
        bag_class = "unknown"
        csv_path = os.path.join(save_path, bag_class, os.path.basename(bags_list[i])+'.csv')
        
        if os.path.exists(csv_path):
            continue

        feats_list = []
        if magnification=='single' or magnification=='low':
            csv_file_path = [path for path in glob.glob(os.path.join(bags_list[i], '*.png')) if os.path.normpath(path) in candidate_table_5x]
        else:
            csv_file_path = [path for path in glob.glob(os.path.join(bags_list[i], "*", '*.png')) if os.path.normpath(path) in candidate_table_20x]
        
        print(len(csv_file_path))
        dataloader, bag_size = bag_dataset(args, csv_file_path)
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch.float().cuda(non_blocking=True) 
                feats, classes = i_classifier(patches)
                feats = feats.cpu().numpy()
                feats_list.extend(feats)
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i+1, num_bags, iteration+1, len(dataloader)))
        if len(feats_list) == 0:
            print('No valid patch extracted from: ' + bags_list[i])
        else:
            df = pd.DataFrame(feats_list)
            os.makedirs(os.path.join(save_path, bag_class), exist_ok=True)
            df.to_csv(csv_path, index=False, float_format='%.4f')
  
        
def compute_tree_feats(args, bags_list, embedder_low, embedder_high, save_path=None, fusion='fusion', tumor_patch_txt=None):
    embedder_low.eval()
    embedder_high.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    
    candidate_table_5x, candidate_table_20x = get_tumor_patch_list(tumor_patch_txt)

    with torch.no_grad():
        for i in range(0, num_bags):
            print(bags_list[i])
            bag_class = "unknown"
            csv_path = os.path.join(save_path, bag_class, os.path.basename(bags_list[i])+'.csv')
            
            if os.path.exists(csv_path):
                continue
            
            low_patches = [path for path in glob.glob(os.path.join(bags_list[i], '*.png')) if os.path.normpath(path) in candidate_table_5x]
            
            feats_list = []
            feats_tree_list = []
            dataloader, bag_size = bag_dataset(args, low_patches)
            for iteration, batch in enumerate(dataloader):
                patches = batch.float().cuda(non_blocking=True)
                feats, classes = embedder_low(patches)
                feats = feats.cpu().numpy()
                feats_list.extend(feats)
            for idx, low_patch in enumerate(low_patches):
                high_folder = os.path.join(os.path.dirname(low_patch), os.path.splitext(os.path.basename(low_patch))[0])
                # high_folder = os.path.splitext(low_patch)[0]
                
                high_patches = [path for path in glob.glob(os.path.join(high_folder, '*.png')) if os.path.normpath(path) in candidate_table_20x]
                    
                
                if not high_patches: # Empty
                    pass
                else:
                    for high_patch in high_patches:
                        try:
                            img = Image.open(high_patch)
                            img = VF.to_tensor(img).float().cuda()
                            feats, classes = embedder_high(img[None, :])
                            if fusion == 'fusion':
                                feats = feats.cpu().numpy()+0.25*feats_list[idx]
                            if fusion == 'cat':
                                feats = np.concatenate((feats.cpu().numpy(), feats_list[idx][None, :]), axis=-1)
                            feats_tree_list.extend(feats)
                        except:
                            os.remove(high_patch)
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i+1, num_bags, idx+1, len(low_patches)))
            if len(feats_tree_list) == 0:
                print('No valid patch extracted from: ' + bags_list[i])
            else:
                df = pd.DataFrame(feats_tree_list)
                os.makedirs(os.path.join(save_path, bag_class), exist_ok=True)
                df.to_csv(csv_path, index=False, float_format='%.4f')
            print('\n')            



def save_log(t):
    with open("log.txt", "w") as f:
        f.write(t)


def compute_features_main(mpps):
    parser = argparse.ArgumentParser(description='Compute TCGA features from SimCLR embedder')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size of dataloader [128]')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of threads for datalodaer')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--backbone', default='resnet18', type=str, help='Embedder backbone [resnet18]')
    parser.add_argument('--norm_layer', default='instance', type=str, help='Normalization layer [instance]')
    parser.add_argument('--magnification', default='tree', type=str, help='Magnification to compute features. Use `tree` for multiple magnifications. Use `high` if patches are cropped for multiple resolution and only process higher level, `low` for only processing lower level.')
    parser.add_argument('--weights', default="None", type=str, help='Folder of the pretrained weights, simclr/runs/*')
    parser.add_argument('--weights_high', default="c16_20x", type=str, help='Folder of the pretrained weights of high magnification, FOLDER < `simclr/runs/[FOLDER]`')
    parser.add_argument('--weights_low', default="c16_5x", type=str, help='Folder of the pretrained weights of low magnification, FOLDER <`simclr/runs/[FOLDER]`')
    # parser.add_argument('--weights_low', default="Oct06_18-21-06_ubuntu", type=str, help='Folder of the pretrained weights of low magnification, FOLDER <`simclr/runs/[FOLDER]`')
    args, _ = parser.parse_known_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    args.dataset = datetime.now().strftime("%y%m%d")
    save_log(args.dataset)
    
    BASE = "./Tiles"
    TUMOR_PATCH_TXT = ("./pseudo-invasive-cancer-patches_5x.txt", "./pseudo-invasive-cancer-patches_20x.txt")
    if args.norm_layer == 'instance':
        norm=nn.InstanceNorm2d
        pretrain = False
    elif args.norm_layer == 'batch':  
        norm=nn.BatchNorm2d
        if args.weights == 'ImageNet':
            pretrain = True
        else:
            pretrain = False

    if args.backbone == 'resnet18':
        resnet = models.resnet18(pretrained=pretrain, norm_layer=norm)
        num_feats = 512
    if args.backbone == 'resnet34':
        resnet = models.resnet34(pretrained=pretrain, norm_layer=norm)
        num_feats = 512
    if args.backbone == 'resnet50':
        resnet = models.resnet50(pretrained=pretrain, norm_layer=norm)
        num_feats = 2048
    if args.backbone == 'resnet101':
        resnet = models.resnet101(pretrained=pretrain, norm_layer=norm)
        num_feats = 2048
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()
    
    
    
    for magnification in ["tree", "high", "low"]:
    # for magnification in ["high"]:
        args.magnification = magnification
        if args.magnification == 'tree' and args.weights_high != None and args.weights_low != None:
            i_classifier_h = mil.IClassifier(resnet, num_feats, output_class=args.num_classes).cuda()
            i_classifier_l = mil.IClassifier(copy.deepcopy(resnet), num_feats, output_class=args.num_classes).cuda()
            
            if args.weights_high == 'ImageNet' or args.weights_low == 'ImageNet' or args.weights== 'ImageNet':
                if args.norm_layer == 'batch':
                    print('Use ImageNet features.')
                else:
                    raise ValueError('Please use batch normalization for ImageNet feature')
            else:
                weight_path = os.path.join('weights', "simclr", 'model-20x.pth')
                state_dict_weights = torch.load(weight_path)
                for i in range(4):
                    state_dict_weights.popitem()
                state_dict_init = i_classifier_h.state_dict()
                new_state_dict = OrderedDict()
                for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                    name = k_0
                    new_state_dict[name] = v
                i_classifier_h.load_state_dict(new_state_dict, strict=False)

                weight_path = os.path.join('weights', "simclr", 'model-5x.pth')
                state_dict_weights = torch.load(weight_path)
                for i in range(4):
                    state_dict_weights.popitem()
                state_dict_init = i_classifier_l.state_dict()
                new_state_dict = OrderedDict()
                for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                    name = k_0
                    new_state_dict[name] = v
                i_classifier_l.load_state_dict(new_state_dict, strict=False)
                print('Use pretrained features.')


        elif args.magnification == 'single' or args.magnification == 'high' or args.magnification == 'low':  
            i_classifier = mil.IClassifier(resnet, num_feats, output_class=args.num_classes).cuda()

            if args.weights == 'ImageNet':
                if args.norm_layer == 'batch':
                    print('Use ImageNet features.')
                else:
                    print('Please use batch normalization for ImageNet feature')
            else:
                if args.magnification == 'high':
                    weight_path = os.path.join('weights', "simclr", 'model-20x.pth')
                else:
                    weight_path = os.path.join('weights', "simclr", 'model-5x.pth')
                    
                state_dict_weights = torch.load(weight_path)
                for i in range(4):
                    state_dict_weights.popitem()
                state_dict_init = i_classifier.state_dict()
                new_state_dict = OrderedDict()
                for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                    name = k_0
                    new_state_dict[name] = v
                i_classifier.load_state_dict(new_state_dict, strict=False)
                print('Use pretrained features.')
        
        
        if args.magnification == 'tree' or args.magnification == 'low' or args.magnification == 'high' :
            bags_path = os.path.join(BASE, 'pyramid')
                
        else:
            bags_path = os.path.join(BASE, 'single')
        feats_path = os.path.join('datasets', args.magnification, f"{args.dataset}")

        os.makedirs(feats_path, exist_ok=True)
        # bags_list = glob.glob(bags_path)
        bags_list = []
        for sid in mpps.keys():
            bags_list.append(os.path.join(bags_path, sid))

        if args.magnification == 'tree':
            compute_tree_feats(args, bags_list, i_classifier_l, i_classifier_h, feats_path, 'cat', TUMOR_PATCH_TXT)
        else:
            compute_feats(args, bags_list, i_classifier, feats_path, args.magnification, TUMOR_PATCH_TXT)
            
            
            
        n_classes = glob.glob(os.path.join(feats_path, '*'+os.path.sep))
        n_classes = sorted(n_classes)
        all_df = []
        for i, item in enumerate(n_classes):
            for sid in mpps.keys():
                bag_csvs = glob.glob(os.path.join(item, f'{sid}.csv'))
                bag_df = pd.DataFrame(bag_csvs)
                bag_df['label'] = i
                bag_df.to_csv(os.path.join(feats_path, item.split(os.path.sep)[2]+'.csv'), index=False)
                all_df.append(bag_df)
        bags_path = pd.concat(all_df, axis=0, ignore_index=True)
        bags_path = shuffle(bags_path)
        bags_path.to_csv(os.path.join(feats_path, f"{args.dataset}"+'.csv'), index=False)
