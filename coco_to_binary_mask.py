#to put in folder testX
import os
import argparse
import sys


from pycocotools.coco import COCO
from matplotlib import image
from pathlib import Path
import numpy as np
from re import findall

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="./dataset")
    args = parser.parse_args()
    #TODO: differentiate masks of different categories

    dataset_path = args.dataset_path
    masks_path = os.path.join(dataset_path, "masks")
    if not os.path.exists(masks_path):
        os.makedirs(masks_path)
    annFile =  os.path.join(dataset_path, "annotations.json")

    coco = COCO(annFile)

    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    annsIds = coco.getAnnIds()

    for imgId in imgIds:
        img = coco.loadImgs(imgId)[0]
        width = coco.imgs[imgId]["width"]
        height = coco.imgs[imgId]["height"]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        img_id = findall(r'\d+', img["file_name"])[0]

        mask = np.zeros((height, width))

        try: 
            mask = np.zeros(coco.annToMask(anns[0]).shape)
            for ann in anns:
                mask += coco.annToMask(ann)
            mask[mask >= 1] = 1
        except:
            pass

        mask_png_name = "mask" +str(img_id) + ".png"
        mask_png_path = os.path.join(masks_path, mask_png_name)
        image.imsave(mask_png_path, mask, cmap='gray')


