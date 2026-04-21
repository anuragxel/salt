from pycocotools import mask
from pycocotools.coco import COCO
from skimage import measure
import json
import shutil
import itertools
import numpy as np
from simplification.cutil import simplify_coords_vwp, simplify_coords
import os, cv2, copy
from distinctipy import distinctipy

from datetime import datetime


def init_coco(dataset_folder, image_names, categories):
    """
    Initialize COCO format JSON.

    :param dataset_folder: Path to the dataset folder.
    :param image_names: List of image names.
    :param categories: List of category names.

    :return: dictionary representing the COCO JSON structure.
    """
    coco_json = {
        "info": {
            "description": "SAM Dataset",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Sam",
            "date_created": datetime.now().strftime(r"%Y/%m/%d"),
        },
        "images": [],
        "annotations": [],
        "categories": [],
    }
    for i, category in enumerate(categories):
        coco_json["categories"].append({"id": i, "name": category, "supercategory": category})
    for i, image_name in enumerate(image_names):
        im = cv2.imread(os.path.join(dataset_folder, image_name))
        coco_json["images"].append(
            {
                "id": i,
                "file_name": image_name,
                "width": im.shape[1],
                "height": im.shape[0],
            }
        )

    return coco_json


def bunch_coords(coords):
    """
    Convert a flat list of coordinates into a list of coordinate pairs.

    :param coords: List of coordinates.

    :return: List of coordinate pairs.
    """
    coords_trans = []
    for i in range(0, len(coords) // 2):
        coords_trans.append([coords[2 * i], coords[2 * i + 1]])
    return coords_trans


def unbunch_coords(coords):
    """
    Convert a list of coordinate pairs into a flat list of coordinates.

    :param coords: List of coordinate pairs.

    :return: Flat list of coordinates.
    """
    return list(itertools.chain(*coords))


def bounding_box_from_mask(mask):
    """
    Calculate the bounding box from a binary mask.

    :param mask: Binary mask (numpy array).

    :return: Tuple (x, y, width, height) representing the bounding box.
    """
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = []
    for contour in contours:
        all_contours.extend(contour)
    convex_hull = cv2.convexHull(np.array(all_contours))
    x, y, w, h = cv2.boundingRect(convex_hull)
    return x, y, w, h


def parse_mask_to_coco(image_id, anno_id, image_mask, category_id, tracker_id=-1, poly=False):
    """
    Parse a binary mask to COCO format annotation.

    :param image_id: ID of the image.
    :param anno_id: ID of the annotation.
    :param image_mask: Binary mask (numpy array).
    :param category_id: ID of the category.
    :param tracker_id: (optional) ID for tracking tasks.
    :param poly: (optional) Boolean indicating whether to use polygon segmentation.

    :return: Dictionary representing the COCO annotation.
    """
    start_anno_id = anno_id
    x, y, width, height = bounding_box_from_mask(image_mask)
    if poly == False:
        fortran_binary_mask = np.asfortranarray(image_mask)
        encoded_mask = mask.encode(fortran_binary_mask)
    if poly == True:
        contours, _ = cv2.findContours(image_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    annotation = {
        "id": start_anno_id,
        "image_id": image_id,
        "tracker_id": tracker_id,
        "category_id": category_id,
        "bbox": [float(x), float(y), float(width), float(height)],
        "area": float(width * height),
        "iscrowd": 0,
        "segmentation": [],
    }
    if poly == False:
        annotation["segmentation"] = encoded_mask
        annotation["segmentation"]["counts"] = str(annotation["segmentation"]["counts"], "utf-8")
    if poly == True:
        for contour in contours:
            sc = contour.ravel().tolist()
            if len(sc) > 4:
                annotation["segmentation"].append(sc)
    return annotation


class DatasetExplorer:
    def __init__(self, dataset_folder, coco_json_path, categories=None):
        """
        Initialize the DatasetExplorer class.
        - LOAD AN EXISTING DATASET: To load an existing dataset, provide the dataset_folder and a valid coco_json_path.
            example: DatasetExplorer(dataset_folder, coco_json_path)
        - CREATE A NEW DATASET: If the COCO JSON file does not exist, categories must be provided to initialize it.
          In this case, the images shall be in a subfolder named "images" within the dataset_folder.
            example: DatasetExplorer(dataset_folder, coco_json_path, categories)

        :param dataset_folder: Path to the dataset folder.
        :param coco_json_path: Path to the COCO JSON file.
        :param categories: (optional) List of category names.
        """
        self.dataset_folder = dataset_folder
        self.coco_json_path = coco_json_path

        # Create a new COCO JSON if it does not exist
        if not os.path.exists(coco_json_path):
            self.__init_coco_json(categories)

        # Load existing COCO JSON
        self.coco = COCO(coco_json_path)
        self.coco_json = self.coco.dataset

        self.categories = [cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())]

        ids = self.coco.getAnnIds()
        self.global_annotation_id = max(ids) + 1 if ids else 0

        self.__createIndex()
        self.__init_colors()

    def __init_coco_json(self, categories):
        """
        Load and sort the images in the dataset folder, and initialize the COCO JSON structure.
        If categories are not provided, an empty COCO JSON will be created.

        :param categories: List of category names.
        """
        image_names = os.listdir(os.path.join(self.dataset_folder, "images"))
        image_names = [os.path.split(name)[1] for name in image_names if name.endswith(".jpg") or name.endswith(".png")]
        image_names.sort()
        image_names = [os.path.join("images", name) for name in image_names]
        self.coco_json = init_coco(self.dataset_folder, image_names, categories)
        self.save_annotation()

    def __init_colors(self):
        """
        Initialize distinct colors for each category.
        """
        self.__category_colors = distinctipy.get_colors(len(self.categories))
        self.__category_colors = [tuple([int(255 * c) for c in color]) for color in self.__category_colors]

    def __createIndex(self):
        """
        Extension of the COCO createIndex method to also create mappings for tracker IDs.
        This method creates two dictionaries:
        - _trackIdToImg: Maps tracker IDs to lists of image IDs.
        - _trackIdtoAnn: Maps tracker IDs to lists of annotation IDs.
        This allows for quick access to images and annotations based on tracker IDs.
        """
        self.coco.createIndex()
        self._trackIdToImg = {}
        self._trackIdtoAnn = {}

        for ann in self.coco_json["annotations"]:
            tracker_id = ann.get("tracker_id", -1)
            if tracker_id < 0:
                continue
            if tracker_id not in self._trackIdToImg:
                self._trackIdToImg[tracker_id] = []
            if tracker_id not in self._trackIdtoAnn:
                self._trackIdtoAnn[tracker_id] = []

            self._trackIdToImg[tracker_id].append(ann["image_id"])
            self._trackIdtoAnn[tracker_id].append(ann["id"])

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None, trackId=-1):
        """
        Extension of the COCO getAnnIds method to also filter by tracker ID.
        It takes the same parameters as the original method, with an additional trackId parameter.
        If trackId is provided, it filters the annotations based on the tracker ID.
        """
        annIds = self.coco.getAnnIds(imgIds, catIds, areaRng, iscrowd)

        if trackId < 0:
            return annIds

        if not hasattr(self, "_trackIdtoAnn"):
            self.__createIndex()

        if trackId not in self._trackIdtoAnn:
            return []

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            return self._trackIdtoAnn[trackId]

        return [ann for ann in self._trackIdtoAnn[trackId] if ann in annIds]

    def getImgIds(self, catIds=[], areaRng=[], trackId=-1):
        """
        Extension of the COCO getImgIds method to also filter by tracker ID.
        It takes the same parameters as the original method, with an additional trackId parameter.
        If trackId is provided, it filters the images based on the tracker ID.
        """
        imgIds = self.coco.getImgIds(catIds, areaRng)

        if trackId < 0:
            return imgIds

        if not hasattr(self, "_trackIdToImg"):
            self.__createIndex()

        if trackId not in self._trackIdToImg:
            return []

        if len(catIds) == len(areaRng) == 0:
            return self._trackIdToImg[trackId]

        return [img for img in self._trackIdToImg[trackId] if img in imgIds]

    def get_colors(self, category_id):
        if len(self.__category_colors) != len(self.categories):
            self.__init_colors()
        return self.__category_colors[category_id]

    def get_categories(self, get_colors=False):
        if get_colors:
            return self.categories, self.__category_colors
        return self.categories

    def get_num_images(self):
        return len(self.coco.imgs)

    def get_image_data(self, image_id):
        """
        Get image data including the image itself, its BGR version, and its embedding.

        :param image_id: ID of the image.

        :return: Tuple (image, image_bgr, image_embedding).
        """

        image_name = self.coco.loadImgs(image_id)[0]["file_name"]
        image_path = os.path.join(self.dataset_folder, image_name)
        embedding_path = os.path.join(
            self.dataset_folder,
            "embeddings",
            os.path.splitext(os.path.split(image_name)[1])[0] + ".npy",
        )
        image = cv2.imread(image_path)
        image_bgr = copy.deepcopy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_embedding = np.load(embedding_path)
        return image, image_bgr, image_embedding

    def get_annotations(self, image_id, return_colors=False):
        """
        Get annotations for a specific image ID.

        :param image_id: ID of the image.
        :param return_colors: If True, also return the colors associated with each annotation.

        :return: List of annotations for the image, and optionally their colors.
        """

        anns = self.coco.loadAnns(self.getAnnIds(imgIds=image_id))


        if return_colors:
            cats = [a["category_id"] for a in anns]
            colors = [self.get_colors(c) for c in cats]
            return anns, colors
        return anns

    def delete_annotations(self, annotation_id):
        """
        Delete a specific annotation by its ID for a given image.

        :param annotation_id: ID of the annotation to delete.
        """
        if not isinstance(annotation_id, int):
            raise ValueError("Annotation ID must be an integer.")

        ann = self.coco.loadAnns(annotation_id)[0]

        try:
            self.coco_json["annotations"].remove(ann)
        except ValueError:
            raise ValueError(f"Annotation with ID {annotation_id} does not exist.")

        self.__createIndex()

    def add_annotation(self, image_id, category_id, mask, tracker_id=-1, poly=True):
        """
        Add a new annotation to the COCO JSON.

        :param image_id: ID of the image.
        :param category_id: ID of the category.
        :param mask: Binary mask (numpy array) for the annotation.
        :param tracker_id: (optional) ID for tracking tasks.
        :param poly: If True, use polygon segmentation; otherwise, use binary mask.

        """
        if mask is None:
            return
        annotation = parse_mask_to_coco(
            image_id, self.global_annotation_id, mask, category_id, tracker_id=tracker_id, poly=poly
        )
        self.coco_json["annotations"].append(annotation)
        self.__createIndex()
        self.global_annotation_id += 1

    def save_annotation(self):
        """
        Save the COCO JSON to the specified path.
        """
        with open(self.coco_json_path, "w") as f:
            json.dump(self.coco_json, f)

    def update_annotation_category(self, new_category_id, selected_annotations):
        """
        Update the category ID of selected annotations.
        :param new_category_id: New category ID to set.
        :param selected_annotations: List of annotation IDs to update.
        """
        anns = self.coco.loadAnns(selected_annotations)
        for a in anns:
            a["category_id"] = new_category_id
        self.__createIndex()

    def update_annotation_tracker_id(self, selected_annotations, new_id):
        """
        Update the tracker ID of selected annotations.
        :param selected_annotations: List of annotation IDs to update.
        :param new_id: New tracker ID to set.
        """
        if not isinstance(new_id, int):
            raise ValueError("new_id must be an integer")
        if new_id < 0:
            raise ValueError("new_id must be a non-negative integer")
        if len(selected_annotations) == 0 or len(selected_annotations) > 1:
            raise ValueError("Exactly one annotation must be selected to change its tracker ID")
        sel = self.coco.loadAnns(selected_annotations)[0]

        old_tracker_id = sel.get("tracker_id", -1)
        sel["tracker_id"] = new_id

        # Change all annotations with the same tracker ID for images that have the same tracker ID and image ID greater than selected annotation
        if old_tracker_id >= 0:
            anns = self.coco.loadAnns(self.getAnnIds(trackId=old_tracker_id))
            for ann in anns:
                if ann["image_id"] > sel["image_id"]:
                    ann["tracker_id"] = new_id

        self.__createIndex()