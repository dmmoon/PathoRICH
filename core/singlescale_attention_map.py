import core.dsmil_attention as mil

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
import warnings
from tqdm import tqdm

from config import THRESHOLD_20x, THRESHOLD_5x
from core.compute_features import get_tumor_patch_list
from core.test import load_log

DOWNSCALE = 4
HIGHSCALE_PATCH_SIZE = 224
LOWSCALE_PATCH_SIZE = HIGHSCALE_PATCH_SIZE * 4

class BagDataset():
    def __init__(self, csv_file, transform=None, args=None):
        self.files_list = csv_file
        self.transform = transform
        self.args = args
        
        
    def __len__(self):
        return len(self.files_list)
    
    
    def __getitem__(self, idx):
        path = self.files_list[idx]
        img = Image.open(path)
        
        img_name = os.path.basename(path)
            
        img_pos = np.asarray(list(map(int, os.path.splitext(img_name)[0].split("_"))))
        sample = {'input': img, 'position': img_pos}
        
        if self.transform:
            sample = self.transform(sample)
        return sample 

class ToTensor(object):
    def __call__(self, sample):
        img = sample["input"]
        img = VF.to_tensor(img)
        sample["input"] = img
        return sample
    
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
                                        ToTensor()
                                    ]),
                                    args=args)
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)


def test(args, bags_list, embedder, milnet, tumor_patch_txt):
    embedder.eval()
    milnet.eval()
    num_bags = len(bags_list)
    
    candidate_table_5x, candidate_table_20x = get_tumor_patch_list(tumor_patch_txt)

    result = []
    for i in tqdm(range(0, num_bags), total=num_bags):
        sid = os.path.basename(bags_list[i])
        
        feats_list = torch.Tensor().cuda()
        pos_list = []
        
        csv_file_path = []
                
        if args.scale == "5x":
            patch_path = [path for path in glob.glob(os.path.join(bags_list[i], f'*.{args.patch_ext}')) if os.path.normpath(path) in candidate_table_5x]
            # PATCH_SIZE = LOWSCALE_PATCH_SIZE
        else:
            patch_path = [path for path in glob.glob(os.path.join(bags_list[i], "*", '*.png')) if os.path.normpath(path) in candidate_table_20x]
            # PATCH_SIZE = HIGHSCALE_PATCH_SIZE

        for path in glob.glob(patch_path):
            # x, y = list(map(int, os.path.basename(path)[:-4].split("_")))
            csv_file_path.append(path)
        
        dataloader, bag_size = bag_dataset(args, csv_file_path)
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda()
                patch_pos = batch['position']
                
                feats, _ = embedder(patches)
                
                feats_list = torch.concat((feats_list, feats), dim=0)
                
                pos_list.extend(patch_pos.detach().cpu().numpy().tolist())
                

            pos_arr = pos_list

            ins_prediction, bag_prediction, _, _, A = milnet(feats_list)
            max_prediction, _ = torch.max(ins_prediction, 0)  

            test_prediction = (
                (
                    0.5 * torch.softmax(max_prediction, dim=0)
                    + 0.5 * torch.softmax(bag_prediction, dim=1)
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            
            pred_class = ""
            eps = 1e-9
            num_pos_classes = 0 + eps
            flag = True
            for c in range(args.num_classes):          
                if test_prediction[c] >= args.thres[c]:
                    if flag: # first class detected
                        print(bags_list[i] + ' is detected as: ' + args.class_name[c])
                        pred_class += args.class_name[c]
                        # colored_tiles = np.matmul(attentions[:, None], colors[c][None, :])
                    else:
                        print('and ' + args.class_name[c])          
                        pred_class += args.class_name[c]
                        # colored_tiles = colored_tiles + np.matmul(attentions[:, None], colors[c][None, :])
                    flag = False # set flag
                       
            
            if pred_class == "": # Flag does not activate
                c = np.argmax(test_prediction)
                pred_class += args.class_name[c]

            slide_name = bags_list[i].split(os.sep)[-1]
            print(args.thres)
            print(test_prediction)
            result.append((slide_name, test_prediction[0], test_prediction[1], pred_class))
            if args.export_scores:
                df_scores = pd.DataFrame(A.cpu().numpy())
                pos_arr_str = [str(s) for s in pos_arr]
                
                df_scores['pos'] = pos_arr_str
                to_csv_base = os.path.join(args.score_path, pred_class)
                os.makedirs(to_csv_base, exist_ok=True)
                df_scores.to_csv(os.path.join(to_csv_base, slide_name+'.csv'), index=False)
                
                
def singlescale_attention_map_main(mpps):
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description='Testing workflow includes attention computing and color map production')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size of feeding patches')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--feats_size', type=int, default=512)
    parser.add_argument('--thres', nargs='+', type=float, default=[0.5308])
    parser.add_argument('--class_name', nargs='+', type=str, default=["favorable", "poor"])
    parser.add_argument('--embedder_weights', default="embedder/Ovarian_Resnet18_SSL_CAMELYON/embedder-high.pth", type=str, help='Folder of the pretrained weights, simclr/runs/*')
    parser.add_argument('--aggregator_weights', type=str, default='')
    parser.add_argument('--bag_path', type=str, default='./Tiles/pyramid')
    parser.add_argument('--patch_ext', type=str, default='png')
    parser.add_argument('--export_scores', type=int, default=1)
    parser.add_argument('--scale', type=str, default='5x')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    args, _ = parser.parse_known_args()
    
    args.process_time = load_log()
    
    for mag, emb_weight, aggr_weight, thres, in [
        ("20x", "./weights/embedder/embedder-high.pth", "weights/mil/IDC-20x.pth", THRESHOLD_20x)
    ]:
        args.scale = mag
        args.embedder_weights = emb_weight
        args.aggregator_weights = aggr_weight
        args.thres = thres
            
        csv_path = os.path.join("./Result", args.process_time, "Patch Attention Score(CSV)", mag)
        args.score_path = csv_path
        
        TUMOR_PATCH_TXT = ("pseudo-invasive-cancer-patches_5x.txt", "pseudo-invasive-cancer-patches_20x.txt")
        
        resnet = models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)
        for param in resnet.parameters():
            param.requires_grad = False
        resnet.fc = nn.Identity()
        embedder = mil.IClassifier(resnet, args.feats_size, output_class=args.num_classes).cuda()
        msg = embedder.load_state_dict(torch.load(args.embedder_weights), strict=False)
        print(msg)

        b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes).cuda()
        i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()

        state_dict_weights = torch.load(args.aggregator_weights) 
        msg = milnet.load_state_dict(state_dict_weights)
        print(msg)

        # bags_list = glob.glob(os.path.join(args.bag_path, '*'))
        bags_list = []
        for sid in mpps.keys():
            bags_list.append(os.path.join(args.bag_path, sid))
            
        if args.export_scores:
            os.makedirs(args.score_path, exist_ok=True)
        if args.class_name == None:
            args.class_name = ['class {}'.format(c) for c in range(args.num_classes)]
        if len(args.thres) != args.num_classes:
            raise ValueError('Number of thresholds does not match classes.')


        test(args, bags_list, embedder, milnet, TUMOR_PATCH_TXT)