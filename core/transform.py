import numpy as np
import cv2

import numpy as np

import albumentations as A
import albumentations.pytorch.transforms as transforms

import torchvision
import torchstain


T = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: x*255)
])

def execute_stain_normalization(image, **kwargs):
    t_to_transform = T(image)
    norm, _, _ = normalizer.normalize(I=t_to_transform)
    return norm.detach().cpu().numpy().astype(np.uint8)



def load_stain_normalizer():
    path = "core/TMP_IMAGE.png"
    target_image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    global normalizer
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    normalizer.fit(T(target_image))

def get_inference_transform(opt):
    
    load_stain_normalizer()

    transform = [
        A.Lambda(name="Lambda", image=execute_stain_normalization, p=1.0),
        A.Resize(opt.resize_shape, opt.resize_shape, p=1),
        transforms.ToTensorV2(),
    ]
    return A.Compose(transform, additional_targets={"image0": "image", "image1": "image"})
