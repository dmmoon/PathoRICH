# Histopathologic Image–based Deep Learning Classifier for Predicting Platinum-based Treatment Responses in High-grade Serous Ovarian Cancer

## Overview
```
This is the primary code repository for reproducing Pathologic Risk Classifier for high grade serous ovarian carcinoma(HGSOC) (PathoRiCH), a histopathologic image–based treatment response analyzer.
```

## Installation
```
    1. pip install -r requirements.txt
    2. install Qupath 0.3.0 (to check the attention map(.geojson), run in Windows OS)
```

## Instructions, how to inference the Whole Slide Images(WSIs)
```
    1. Move WSIs that you want to analyze to "Input" directory
    2. Run PathoRICH.sh
```


## Running process and outputs (Takes about 20 minutes below per each WSI)
1. Patches of each slide will be generated. The slide which tiling process is done will be moved to *Done* directory automatically.
    - Patches will be saved to *Tiles* directory
    
2. Invasive cancer patches will be classified among the each generated 5x, 20x patch
    - classified cancer patch list will be recored to *pseudo-invasive-cancer-patches-5x.txt* and *pseudo-invasive-cancer-patches-20x.txt*
    - Text files will be removed after the process is completed.

3. Self-supervised(SimCLR) CNN model extract features of each patch that is selected as a cancer patch among the cancer patch list.

4. 5x/20x/multiscale PathoRICH will analyze each WSIs. The **results** will be saved to *Result* directory:
    - The probability of favorable and poor will be saved to csv *(path : Result/{Date}/Slide-level Prediction/)*
        - high.csv : **(Expected Result)** the result of 20x PathoRICH model
        - low.csv : the result of 5x PathoRICH model
        - tree.csv : the result of multiscale PathoRICH
    - Visualization of attention map *(path : Result/{Date}/{magnification}/Geojson)*
        - you can check the attention map via uploading to Qupath
            - 1\) open WSI on Qupath
            - 2\) upload geojson file
    - the clustering result of Attention Patch *(path : Result/{Date}/{magnification}/Geojson)*
        - the high quality of clustering result will be derived if you analyze many WSIs at once.


## Configuration ( config.py )
- Change the value below if you want to get difference results
```
# Number of cluster
N_CLUSTER = 5 (default)

# Size of the figure of clustering map (Width, Height)
figsize = (7, 7) (default)

# Threshold value for classifying cancer patches among the whole 5x patches
LOWSCALE_CANCER_THRESHOLD = 0.2 (default)

# Threshold value for classifying cancer patches among the whole 20x patches
HIGHSCALE_CANCER_THRESHOLD = 0.5 (default)

# Value for choosing high/low N% of attention score
TOP_SCORE_RATIO = 0.05 # (default)
```

## License

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg