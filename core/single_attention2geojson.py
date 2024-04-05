import os
import glob
import shutil
import geojson

import pandas as pd


from tqdm import tqdm

import argparse

from core.test import load_log

from config import TOP_SCORE_RATIO

TUMOR_LIST_PATH = "pseudo-invasive-cancer-patches_20x.txt"
PATCH_BASE_PATH = "./Tiles/pyramid" # /path/to/pyramid

parser = argparse.ArgumentParser(description='')
parser.add_argument('--name', default="", type=str, help='')
args, _ = parser.parse_known_args()

args.name = load_log()
score_name = args.name

SAVE_PATH = f"./Result/{args.name}/Attention Patches"
GEOJSON_PATH = f"./Result/{args.name}/Geojson"

def attention_to_color(score):
    """
    Convert attention score to color scale. Jet colormap is used. 
    """
    palette = {
        0:-16777216,
        10:-16776961,
        20:-10059546,
        30:-3342337,
        40:-16711681,
        50:-16711936,
        60:-256,
        70:-19610,
        80:-3381709,
        90:-6750208,
        100:-65536,
        "favorable": -16776961,
        "poor":-65536
    }
    
    return palette[score]



def export2json(positions, save_path, til_size):
    features = []
    for score, (x, y) in positions:
        color = attention_to_color(score)
        
        xmin = x * til_size
        ymin = y * til_size
        xmax = xmin + til_size
        ymax = ymin + til_size
        
        properties = {
            "object_type": "annotation",
            "classification":{
                "name": score,
                "colorRGB": color
            },
            "isLocked": False
        }
        
        polygon = [
            [[xmin, ymin],
             [xmax, ymin],
             [xmax, ymax],
             [xmin, ymax],
             [xmin, ymin]],
        ]        

        polygon = geojson.Polygon(polygon)
        feature = geojson.Feature(geometry=polygon, properties=properties)
        features.append(feature)        


    feature_collection = geojson.FeatureCollection(features)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        geojson.dump(feature_collection, f, indent=2)




score_table = {
    "favorable": list(),
    "poor": list()
}

def split_patch(csv_path, min_value, max_value, mag, til_size):
    sid = os.path.basename(os.path.splitext(csv_path)[0])
    raw_df = pd.read_csv(csv_path)
    
    pred_class = os.path.basename(os.path.dirname(csv_path))
    
    raw_df["0"] = raw_df["0"].apply(
        lambda x: round((float(x) - min_value) / (max_value - min_value), 2)
    )
    raw_df["1"] = raw_df["1"].apply(
        lambda x: round((float(x) - min_value) / (max_value - min_value), 2)
    )
    
    
    pos_all = []
    for idx, (favorable_score, poor_score) in enumerate(list(zip(raw_df["0"], raw_df["1"]))):
        df = raw_df.copy()
        pos = df['pos'][idx][1:-1].replace(", ", "_")
        if favorable_score >= poor_score:
            s = "favorable"
        else:
            s = "poor"
        pos_all.append((s, list(map(int, pos.split("_")))))
        
        if s == "favorable":
            score_table["favorable"].append((sid+"--"+pos, favorable_score))
        else:
            score_table["poor"].append((sid+"--"+pos, poor_score))

    export2json(pos_all, save_path=os.path.join(GEOJSON_PATH, mag, pred_class, sid+f"--all.geojson"), til_size=til_size)
        
    
    
    for idx, score in enumerate([raw_df["0"], raw_df["1"]]):
        df = raw_df.copy()       
        positions = []
        
        for i in range(len(df)):
            s = int(round(score[i] * 100, -1))
            pos = df["pos"][i][1:-1].replace(", ", "_")
            
            positions.append((s, list(map(int, pos.split("_")))))

            
        if idx == 0:
            att = "favorable"
        else:
            att = "poor"
        export2json(positions, save_path=os.path.join(GEOJSON_PATH, mag, pred_class, sid+f"--{att}.geojson"), til_size=til_size)
    print(sid, "Done")
        
    


def calculate_min_max_value(csv_path):
    df = pd.read_csv(csv_path)

    min_value = min(df[["0", "1"]].min())
    max_value = max(df[["0", "1"]].max())
    
    return min_value, max_value
    
    


def single_attention2geojson_main(mpps):
    import time
    s = time.time()
    
    for sid, mpp in mpps.items():
        for mag in ["20x"]:
            if mpp >= 0.5:
                TIL_SIZE = 896 if mag == "5x" else 224
            else:
                TIL_SIZE = 896 * 2 if mag == "5x" else 224 * 2
            
            attention_csv = glob.glob(f"./Result/{args.name}/Patch Attention Score(CSV)/{mag}/*/{sid}.csv")
            if attention_csv:
                attention_csv = attention_csv[0]
            else:
                continue

            min_value, max_value = calculate_min_max_value(attention_csv)
            split_patch(attention_csv, min_value, max_value, mag, TIL_SIZE)
            
    for flag, dtype in [(True, "high score"), (False, "low score")]:
        for cls_name in score_table.keys():
            print(flag, dtype, cls_name, len(score_table[cls_name]))
            save_base = os.path.join(SAVE_PATH, mag, cls_name, dtype)
            os.makedirs(save_base, exist_ok=True)
            topk = sorted(score_table[cls_name], key=lambda x: x[1], reverse=flag)
            if len(topk) > 100:
                topk = topk[:int(len(score_table[cls_name]) * TOP_SCORE_RATIO)]
            for idx, (pos, score) in enumerate(topk):
                if score >= 0.5:
                    sid, pos = pos.split("--")
                    pp = pos.split("_")
                    target_path = os.path.join(save_base, f"{idx+1}.png")
                    if mag == "5x":
                        patch_path = os.path.join(PATCH_BASE_PATH, sid, pos+'.png')
                    else:
                        patch_path = os.path.join(PATCH_BASE_PATH, sid, f"{int(pp[0])//4}_{int(pp[1])//4}", pos+'.png')
                    
                    try:
                        shutil.copy(patch_path, target_path)
                    except Exception as e:
                        pass
    # score_table["favorable"].clear()
    # score_table["poor"].clear()
        
    shutil.rmtree(f"./Result/{args.name}/Patch Attention Score(CSV)")
