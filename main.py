import os
import shutil
import argparse


def reset():
    # shutil.rmtree("Tiles/pyramid")
    os.remove("pseudo-invasive-cancer-patches_5x.txt")
    os.remove("pseudo-invasive-cancer-patches_20x.txt")
    os.remove("log.txt")
    shutil.rmtree("datasets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    _ = parser.parse_known_args()
    from core.deepzoom_tiler import deepzoom_tiler_main
    mpps = deepzoom_tiler_main()
    from core.invasive_cancer_pseudo_inference import invasive_cancer_pseudo_inference_main
    tumor_areas = invasive_cancer_pseudo_inference_main(mpps)
    from core.compute_features import compute_features_main
    compute_features_main(mpps)
    from core.test import test_main
    test_main()
    from core.singlescale_attention_map import singlescale_attention_map_main
    singlescale_attention_map_main(mpps)
    from core.single_attention2geojson import single_attention2geojson_main
    single_attention2geojson_main(mpps)
    from core.clustering import clustering_main
    clustering_main()
    reset()
    print("---Analyze Complete---")
