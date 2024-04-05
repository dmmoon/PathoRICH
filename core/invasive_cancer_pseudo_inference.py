import os
import sys
import glob
import torch
import argparse

import numpy as np
import torch.backends.cudnn as cudnn
import segmentation_models_pytorch as smp

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {'0', '1', '2'}

DATA_PATH = f"./Tiles/pyramid"

from config import LOWSCALE_CANCER_THRESHOLD, HIGHSCALE_CANCER_THRESHOLD
# THRESHOLD = 0.2

SAVE_TXT_PATH = f"pseudo-invasive-cancer-patches"

import random

torch.manual_seed(32)
torch.cuda.manual_seed(32)
np.random.seed(32)
random.seed(32)
cudnn.deterministic = True
cudnn.benchmark = False



def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--txt_path', type=str, default=SAVE_TXT_PATH,
                        help='num of workers to use')
    
    # model dataset
    parser.add_argument('--activation', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help='path to custom dataset')

    # other setting
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--parallel', action="store_true",
                        help='Distributed Parellel Training')
    parser.add_argument('--stain_normalization', type=str, default=None)


    opt, _ = parser.parse_known_args()
    opt.model_path = "./weights/IDC-Segmentation/IDC-Segmentation.pt"
    opt.image_size = 224
    
    return opt


class InvasiveCancerSegmentationDataset(Dataset):
    def __init__(
            self, 
            data_paths,
            data,
            preprocessing=None
    ):
        self.data = data
        self.images_fps = self.get_data_paths(data_paths)
        self.preprocessing = preprocessing


    def get_data_paths(self, base_path):
        ret = []
        for sid in self.data:
            ret += glob.glob(os.path.join(base_path, sid, "*.png"))
        return ret
                
    
    def __getitem__(self, i):
        try:
            image = Image.open(self.images_fps[i])
        except:
            os.remove(self.images_fps[i])
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        
        if self.preprocessing:
            image = self.preprocessing(image)
        
        return image
        

    def __len__(self):
        return len(self.images_fps)



class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    



def get_transforms(**kwopt):

    transform = [
        transforms.ToTensor()
        
    ]
    return transforms.Compose(transform)


def set_loader(opt, data):
    transform = get_transforms()

    valid_dataset = InvasiveCancerSegmentationDataset(
        data_paths=opt.data_path,
        data=data,
        preprocessing=transform,
    )

    print("Dataset", len(valid_dataset))
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size, shuffle=None,
        num_workers=opt.num_workers, pin_memory=True, sampler=None)

    print("Init Data Loader")
    return valid_loader



def set_model(opt):  
    "invasive cancer segmentation model"

    model = smp.UnetPlusPlus(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        classes=1,
        activation=None
    )

    model.load_state_dict(torch.load(opt.model_path, map_location=f"cuda"))
    model = model.cuda()

    model.eval()

    print("Init Model")
    return model



def validate(mpps, val_loader, model, opt):
    """validation"""
    def run_validate(sids, loader, base_progress=0):
        low_fd = open(opt.txt_path+"_5x.txt", "w")
        high_fd = open(opt.txt_path+"_20x.txt", "w")
        
        
        table = dict()
        for sid in sids:
            table[os.path.splitext(sid)[0]] = 0
        
        with torch.no_grad():
            for idx, images in enumerate(loader):
                idx = base_progress + idx
                images = images.cuda(opt.gpu, non_blocking=True).type(torch.float32)

                output = model(images)

                # output = output.sigmoid().round()
                output = output.sigmoid().squeeze(dim=1).round().detach().cpu().numpy()
                
                for i, o in enumerate(output):
                    ii = idx * opt.batch_size + i
                    fpath = val_loader.dataset.images_fps[ii]
                    # print(np.count_nonzero(o) / (opt.image_size * opt.image_size))
                    # 5x patch
                    ones = np.count_nonzero(o)
                    if ones / (opt.image_size * opt.image_size) >= LOWSCALE_CANCER_THRESHOLD:
                        low_fd.write(f"{os.path.normpath(fpath)},{ones/(opt.image_size*opt.image_size):.2f}\n")
                        
                    table[os.path.basename(os.path.dirname(fpath))] += ones
                        
                    # 20x patch
                    size_20x = opt.image_size // 4
                    if os.path.basename(fpath)[0].isalpha():
                        ftype, location = os.path.basename(fpath).split("-")
                        low_x, low_y = location[:-4].split("_")
                    else:
                        ftype = None
                        low_x, low_y = os.path.basename(fpath)[:-4].split("_")
                        
                    for jj in range(opt.image_size // size_20x):
                        for ii in range(opt.image_size // size_20x):
                            high_patch_area = o[ii*size_20x:(ii+1)*size_20x, jj*size_20x:(jj+1)*size_20x]
                            ones = np.count_nonzero(high_patch_area)
                            if ones / (size_20x * size_20x) >= HIGHSCALE_CANCER_THRESHOLD:
                                if ftype is not None:
                                    high_patch_path = os.path.join(fpath[:-4], f"{ftype}-{int(low_x)*4+(ii)}_{int(low_y)*4+(jj)}.png")
                                else:
                                    high_patch_path = os.path.join(fpath[:-4], f"{int(low_x)*4+(ii)}_{int(low_y)*4+(jj)}.png")
                                    
                                if os.path.exists(high_patch_path):
                                    high_fd.write(f"{os.path.normpath(high_patch_path)},{ones/(size_20x*size_20x):.2f}\n")

                        
                if idx % opt.print_freq == 0:
                    progress.display(idx + 1)
        low_fd.close()
        high_fd.close()
        return table

    progress = ProgressMeter(
        len(val_loader),
        [],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    table = run_validate(list(mpps.keys()), val_loader)
    
    for sid, mpp in mpps.items():
        pixel_tumor_area = table[os.path.splitext(sid)[0]]
        if mpp != -1:
            mpps[sid] = round(pixel_tumor_area * (mpp ** 2) * 1e-6, 2) # 픽셀 단위 -> 제곱밀리미터 단위로 변환
        else:
            mpps[sid] = round(pixel_tumor_area * 1e-6, 2) # 픽셀 단위 -> 제곱밀리미터 단위로 변환
            
    
    progress.display_summary()

    return mpps



def invasive_cancer_pseudo_inference_main(mpps):
    opt = parse_option()

    if opt.gpu is not None:
        print("Use GPU: {} for inference".format(opt.gpu))    

    model = set_model(opt)

    val_loader = set_loader(opt, list(mpps.keys()))

    return validate(mpps, val_loader, model, opt)


