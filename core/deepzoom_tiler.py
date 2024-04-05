# python deepzoom_tiler.py -m 0 -b 10
# python deepzoom_tiler.py -m 0 2 -b 20 

import json
from multiprocessing import Process, JoinableQueue
import argparse
import os
import re
import shutil
import sys
import glob
import numpy as np
import math
from unicodedata import normalize
from skimage import io
from skimage.color import rgb2hsv
from skimage.util import img_as_ubyte
from skimage import filters
from PIL import Image, ImageFilter, ImageStat

import cv2

Image.MAX_IMAGE_PIXELS = None

import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator

import ray

VIEWER_SLIDE_NAME = 'slide'

DOWNSTREAM = 8
NUM_CPU = 4
BLOOD_THRESHOLD = 0.10



def is_in_blood(img, tile_size, threshold=0.09):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    #set the bounds for the red hue
    lower_red = np.array([160,100,50])
    upper_red = np.array([180,255,255])

    #create a mask using the bounds set
    mask = cv2.inRange(hsv, lower_red, upper_red)
    if np.count_nonzero(mask) / (tile_size * tile_size) >= threshold:
        return True
    else:
        return False
    

class TileWorker(Process):
    """A child process that generates and writes tiles."""

    def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds,
                quality, threshold):
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        self._queue = queue
        self._slidepath = slidepath
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._quality = quality
        self._threshold = threshold
        self._slide = None

    def run(self):
        self._slide = open_slide(self._slidepath)
        last_associated = None
        dz = self._get_dz()
        
        
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break
            associated, level, address, outfile = data
            if last_associated != associated:
                dz = self._get_dz(associated)
                last_associated = associated
            try:
                
                # x5 (level 15) = (896, 896), x20 (level 17) = (224, 224)
                tile = dz.get_tile(level, address)
                (_, _, tile_size), _ = dz._get_tile_info(level, address)
                tile_size = tile_size[0]
                                
                col, row = address
                w, h = tile.size
                if not (w==self._tile_size and h==self._tile_size):
                    tile = tile.resize((self._tile_size, self._tile_size))

                edge = tile.filter(ImageFilter.FIND_EDGES)
                edge = ImageStat.Stat(edge).sum
                edge = np.mean(edge)/ (self._tile_size**2)
                    
                if edge > self._threshold:
                    if not is_in_blood(np.array(tile), self._tile_size, threshold=BLOOD_THRESHOLD):
                        outfile = os.path.join(os.path.dirname(outfile), f"{os.path.basename(outfile)}")
                        tile.save(outfile)
                    
            except Exception as e:
                print(e)
            self._queue.task_done()
    

    def _get_dz(self, associated=None):
        if associated is not None:
            image = ImageSlide(self._slide.associated_images[associated])
        else:
            image = self._slide
        return DeepZoomGenerator(image, self._tile_size, self._overlap,
                    limit_bounds=self._limit_bounds)


class DeepZoomImageTiler(object):
    """Handles generation of tiles and metadata for a single image."""

    def __init__(self, dz, basename, target_levels, mag_base, format, associated, queue):
        self._dz = dz
        self._basename = basename
        self._format = format
        self._associated = associated
        self._queue = queue
        self._processed = 0
        self._target_levels = target_levels
        self._mag_base = int(mag_base)

    def run(self):
        self._write_tiles()

    def _write_tiles(self):
        target_levels = [self._dz.level_count-i-1 for i in self._target_levels]
        mag_list = [int(self._mag_base/2**i) for i in self._target_levels]
        mag_idx = 0
        for level in range(self._dz.level_count):
            if not (level in target_levels):
                continue
            tiledir = os.path.join("%s_files" % self._basename, str(mag_list[mag_idx]))
            if not os.path.exists(tiledir):
                os.makedirs(tiledir)
            cols, rows = self._dz.level_tiles[level]
            for row in range(rows):
                for col in range(cols):
                    tilename = os.path.join(tiledir, '%d_%d.%s' % (
                                    col, row, self._format))
                    if not os.path.exists(tilename):
                        self._queue.put((self._associated, level, (col, row),
                                    tilename))
                    self._tile_done()
            mag_idx += 1

    def _tile_done(self):
        self._processed += 1
        count, total = self._processed, self._dz.tile_count
        if count % 100 == 0 or count == total:
            print("Tiling %s: wrote %d/%d tiles" % (
                    self._associated or 'slide', count, total),
                    end='\r', file=sys.stderr)
            if count == total:
                print(file=sys.stderr)


class DeepZoomStaticTiler(object):
    """Handles generation of tiles and metadata for all images in a slide."""

    def __init__(self, slidepath, basename, mag_levels, base_mag, objective, format, tile_size, overlap,
                limit_bounds, quality, workers, threshold):
        self._slide = open_slide(slidepath)
        self.mpp = self._slide.properties.get(openslide.PROPERTY_NAME_MPP_X, -1)
        print(self.mpp)
        self._basename = basename
        self._format = format
        self._tile_size = tile_size
        self._overlap = overlap
        self._mag_levels = mag_levels
        self._base_mag = base_mag
        self._objective = objective
        self._limit_bounds = limit_bounds
        self._queue = JoinableQueue(2 * workers)
        self._workers = workers
        self._dzi_data = {}
        for _i in range(workers):
            TileWorker(self._queue, slidepath, tile_size, overlap,
                        limit_bounds, quality, threshold).start()

    def run(self):
        self._run_image()
        self._shutdown()
        return self.mpp

    def _run_image(self, associated=None):
        """Run a single image from self._slide."""
        if associated is None:
            image = self._slide
            basename = self._basename
        else:
            image = ImageSlide(self._slide.associated_images[associated])
            basename = os.path.join(self._basename, self._slugify(associated))
        dz = DeepZoomGenerator(image, self._tile_size, self._overlap,
                    limit_bounds=self._limit_bounds)
        
        MAG_BASE = self._slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        if MAG_BASE is None:
            MAG_BASE = self._objective
        first_level = int(math.log2(float(MAG_BASE)/self._base_mag)) # raw / input, 40/20=2, 40/40=0
        target_levels = [i+first_level for i in self._mag_levels] # levels start from 0
        target_levels.reverse()
        
        tiler = DeepZoomImageTiler(dz, basename, target_levels, MAG_BASE, self._format, associated,
                    self._queue)
        tiler.run()

    def _url_for(self, associated):
        if associated is None:
            base = VIEWER_SLIDE_NAME
        else:
            base = self._slugify(associated)
        return '%s.dzi' % base

    def _copydir(self, src, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)
        for name in os.listdir(src):
            srcpath = os.path.join(src, name)
            if os.path.isfile(srcpath):
                try:
                    shutil.copy(srcpath, os.path.join(dest, name))
                except:
                    pass

    @classmethod
    def _slugify(cls, text):
        text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
        return re.sub('[^a-z0-9]+', '_', text)

    def _shutdown(self):
        for _i in range(self._workers):
            self._queue.put(None)
        self._queue.join()

def nested_patches(img_slide, out_base, tmp_path, level=(0,), ext='jpeg'):
    print('\n Organizing patches', tmp_path)
    img_name = os.path.splitext(os.path.basename(img_slide))[0]
    bag_path = os.path.join(out_base, img_name)
    
    os.makedirs(bag_path, exist_ok=True)
        
    if len(level)==1:
        patches = glob.glob(os.path.join(tmp_path, '*', '*.'+ext+".*"))
        for i, patch in enumerate(patches):
            patch_name = os.path.basename(patch)
            try:
                shutil.move(patch, os.path.join(bag_path, patch_name))
            except:
                pass
            sys.stdout.write('\r Patch [%d/%d]' % (i+1, len(patches)))
        print('Done.')
    else:
        level_factor = 2**(int(level[1] - level[0]))
        levels = [int(os.path.basename(i)) for i in glob.glob(os.path.join(tmp_path, '*'))]
        levels.sort()
        print(levels)
        low_patches = glob.glob(os.path.join(tmp_path, str(levels[0]), '*.'+ext))
        for i, low_patch in enumerate(low_patches):
            low_patch_name = os.path.basename(low_patch)
            try:
                shutil.move(low_patch, os.path.join(bag_path, low_patch_name)) 
            except:
                pass
            low_patch_folder = low_patch_name.split('.')[0]
            high_patch_path = os.path.join(bag_path, low_patch_folder)
            os.makedirs(high_patch_path, exist_ok=True)
            # low_x = int(low_patch_folder.split('_')[0])
            # low_y = int(low_patch_folder.split('_')[1])
            low_x, low_y = list(map(int, low_patch_folder.split("_")))
            high_x_list = list( range(low_x*level_factor, (low_x+1)*level_factor) )
            high_y_list = list( range(low_y*level_factor, (low_y+1)*level_factor) )
            for x_pos in high_x_list:
                for y_pos in high_y_list:
                    high_patch = os.path.join(tmp_path, str(levels[1]), '{}_{}.'.format(x_pos, y_pos)+ext)
                    if os.path.isfile(high_patch):
                        high_patch_name = os.path.basename(high_patch)
                        try:
                            shutil.move(high_patch, os.path.join(bag_path, low_patch_folder, high_patch_name))
                        except Exception as e:
                            print(e)
            try:
                os.rmdir(os.path.join(bag_path, low_patch_folder))
                os.remove(low_patch)
            except:
                pass
            sys.stdout.write('\r Patch [%d/%d]' % (i+1, len(low_patches)))
        print('Done.')


@ray.remote
def run(args, idx, c_slide, out_base, levels, num_slides):
    sid = os.path.basename(c_slide)
    try:
        print('Process slide {}\t{}/{}'.format(sid, idx+1, num_slides))
        temp_path = f'WSI_temp_{idx}'
        if not os.path.exists(f"Tiles/pyramid/{os.path.splitext(sid)[0]}"):
            mpp = DeepZoomStaticTiler(c_slide, temp_path, levels, args.base_mag, args.objective, args.format, args.tile_size, args.overlap, True, args.quality, args.workers, args.background_t).run()
            nested_patches(c_slide, out_base, temp_path+'_files', levels, ext=args.format)
            shutil.rmtree(f'{temp_path}_files')
        else:
            mpp = open_slide(c_slide).properties.get(openslide.PROPERTY_NAME_MPP_X, -1)
        
        print(sid, mpp)
        try:
            # shutil.move(c_slide, f"Done/{sid}")
            pass
            # pass
        except Exception as e:
            print(e)
        return sid, float(mpp)
    except Exception as e:
        print(e)
        return sid, -1



def initialize():
    for dirname in ["Input", "Done", "datasets", "Result", "Tiles"]:
        if not os.path.isdir(dirname):
            os.makedirs(dirname)



def deepzoom_tiler_main():
    ray.init(num_cpus=2)
    Image.MAX_IMAGE_PIXELS = None
    parser = argparse.ArgumentParser(description='Patch extraction for WSI')
    parser.add_argument('-d', '--dataset', type=str, default='Request', help='Dataset name')
    parser.add_argument('-e', '--overlap', type=int, default=0, help='Overlap of adjacent tiles [0]')
    parser.add_argument('-f', '--format', type=str, default='png', help='Image format for tiles [jpeg]')
    parser.add_argument('-v', '--slide_format', type=str, default='svs', help='Image format for tiles [svs]')
    parser.add_argument('-j', '--workers', type=int, default=4, help='Number of worker processes to start [4]')
    parser.add_argument('-q', '--quality', type=int, default=70, help='JPEG compression quality [70]')
    parser.add_argument('-s', '--tile_size', type=int, default=224, help='Tile size [224]')
    parser.add_argument('-b', '--base_mag', type=float, default=20, help='Maximum magnification for patch extraction [20]')
    parser.add_argument('-m', '--magnifications', type=int, nargs='+', default=(0, 2), help='Levels for patch extraction [0]')
    parser.add_argument('-o', '--objective', type=float, default=20, help='The default objective power if metadata does not present [20]')
    parser.add_argument('-t', '--background_t', type=int, default=10, help='Threshold for filtering background [15]')  
    args, _ = parser.parse_known_args()
    levels = tuple(args.magnifications)
    assert len(levels)<=2, 'Only 1 or 2 magnifications are supported!'
    initialize()
    
    path_base = f"./Input"
    out_base = f"./Tiles"
    out_base = os.path.join(out_base, 'pyramid')
    
    all_slides = glob.glob(os.path.join(path_base, "*.svs"))
    
    pairs = ray.get([run.remote(args, idx, c_slide, out_base, levels, len(all_slides)) for idx, c_slide in enumerate(all_slides)])

    mpps = {os.path.splitext(sid)[0]: mpp for sid, mpp in pairs}
    
    print('Patch extraction done for {} slides.'.format(len(all_slides)))
    ray.shutdown()
    return mpps
