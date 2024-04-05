import torch
from torch.autograd import Variable

import argparse, os, copy
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from config import THRESHOLD_20x, THRESHOLD_5x

from tqdm import tqdm

import random
from torch.backends import cudnn

torch.manual_seed(32)
torch.cuda.manual_seed(32)
np.random.seed(32)
random.seed(32)
cudnn.deterministic = True
cudnn.benchmark = False

def get_bag_feats(csv_file_df, args):
    print(csv_file_df)
    feats_csv_path = csv_file_df.iloc[0]
    df = pd.read_csv(feats_csv_path)
    feats = shuffle(df).reset_index(drop=True)
    feats = feats.to_numpy()
    label = np.zeros(args.num_classes)
    if args.num_classes==1:
        label[0] = csv_file_df.iloc[1]
    else:
        if int(csv_file_df.iloc[1])<=(len(label)-1):
            label[int(csv_file_df.iloc[1])] = 1
        
    return label, feats


def dropout_patches(feats, p):
    idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0]*(1-p)), replace=False)
    sampled_feats = np.take(feats, idx, axis=0)
    pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0]*p), replace=False)
    pad_feats = np.take(sampled_feats, pad_idx, axis=0)
    sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
    return sampled_feats


def test(test_df, milnet, args):
    milnet.eval()
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():
        for i in tqdm(range(len(test_df))):
            label, feats = get_bag_feats(test_df.iloc[i], args)
            bag_label = Variable(Tensor(np.array([label])))
            bag_feats = Variable(Tensor(np.array([feats])))
            bag_feats = bag_feats.view(-1, args.feats_size)
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)  
            # if args.average:
            test_predictions.extend([(0.5*torch.softmax(max_prediction, dim=0)+0.5*torch.softmax(bag_prediction, dim=1)).squeeze().cpu().numpy()])
            # else: test_predictions.extend([(0.0*torch.sigmoid(max_prediction)+1.0*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
    test_sids = test_df["0"].to_numpy()
    test_predictions = np.array(test_predictions)
    test_predictions_prob = test_predictions.copy()
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=args.threshold[0]] = 1
        class_prediction_bag[test_predictions<args.threshold[0]] = 0
        test_predictions = class_prediction_bag
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=args.threshold[i]] = 1
            class_prediction_bag[test_predictions[:, i]<args.threshold[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    
    result2csv(args, test_sids, test_predictions, test_predictions_prob)
                    


def result2csv(args, test_sids, test_predictions, test_predictions_prob):
    if args.dataset == "tree":
        args.dataset == "multiscale"
    csv_path = os.path.join("./Result", args.process_time, "Slide-level Prediction", args.dataset+'.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("Slide ID,Predict Probability(Favorable),Predict Probability(Poor),Predict Class\n")
        for ii, (sid, pred) in enumerate(zip(test_sids, test_predictions)):
            pred_class = ""
            if pred[0] == 1:
                pred_class += "Favorable"
            if pred[1] == 1:
                pred_class += "Poor"
                
            if pred_class == "" or pred_class == "FavorablePoor":
                max_cls_idx = np.argmax(test_predictions_prob[ii])
                if max_cls_idx == 0:
                    pred_class = "Favorable"
                else:
                    pred_class = "Poor"

            f.write(f"{os.path.basename(os.path.splitext(sid)[0])},{test_predictions_prob[ii][0]:.4f},{test_predictions_prob[ii][1]:.4f},{pred_class}\n")   


def load_log():
    with open("log.txt", "r") as f:
        return f.read()


def test_main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=1024, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=8, help='GPU ID(s) [0]')
    parser.add_argument('--dataset', default='', type=str, help='Dataset folder name')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=True, help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--weights', type=str, default="", help='')
    parser.add_argument('--threshold', nargs='+', type=float, default=[0.4041, 0.6221])
    args, _ = parser.parse_known_args()
    
    args.process_time = load_log()
    
    if args.model == 'dsmil':
        import core.dsmil as mil
    
    
    for dataset, feat_size, weight, threshold in [
        ("tree", 1024, "weights/mil/IDC-multiscale.pth", [0.4503, 0.5497]), 
        ("high", 512, "weights/mil/IDC-20x.pth", THRESHOLD_20x), 
        ("low", 512, "weights/mil/IDC-5x.pth", THRESHOLD_5x)
        ]:
        args.dataset = dataset
        args.feats_size = feat_size
        args.threshold = threshold
        args.weights = weight
    
        i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
        b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()
        if args.model == 'dsmil':
            state_dict_weights = torch.load(args.weights)
            try:
                milnet.load_state_dict(state_dict_weights, strict=False)
            except Exception as e:
                print(e)
                del state_dict_weights['b_classifier.v.1.weight']
                del state_dict_weights['b_classifier.v.1.bias']
                milnet.load_state_dict(state_dict_weights, strict=False)
        
        
        bags_csv = os.path.join('datasets', args.dataset, args.process_time, args.process_time+'.csv')
            
        test_path = pd.read_csv(bags_csv)

        print(len(test_path))
        test(test_path, milnet, args)


        