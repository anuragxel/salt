# Segment Anything Labelling Tool (SALT)

Uses the Segment-Anything Model By Meta AI and adds a barebones interface to label images and saves the masks in the COCO format.

Under active development, apologies for rough edges and bugs. Use at your own risk.

## Installation

1. Install [Segment Anything](https://github.com/facebookresearch/segment-anything) on any machine with a GPU. (Need not be the labelling machine.)
2. Create a conda environment using `conda env create -f environment.yaml` on the labelling machine (Need not have GPU).
3. (Optional) Install [coco-viewer](https://github.com/trsvchn/coco-viewer) to scroll through your annotations quickly.

## Usage

1. Setup your dataset in the following format `<dataset_name>/images/*` and create empty folder `<dataset_name>/embeddings`.
    - Annotations will be saved in `<dataset_name>/annotations.json` by default.
2. Copy the `helpers` scripts to the base folder of your `segment-anything` folder.
    - Call `extract_embeddings.py` to extract embeddings for your images.
    - Call `generate_onnx.py` generate `*.onnx` files in models.
4. Copy the models in `models` folder. 
5. Symlink your dataset in the SALT's root folder as `<dataset_name>`.
6. Call `segment_anything_annotator.py` with argument `<dataset_name>` and categories `cat1,cat2,cat3..`.
    - There are a few keybindings that make the annotation process fast.
    - Click on the object using left clicks and right click (to indicate outside object boundary).
    - `n` adds predicted mask into your annotations. (Add button)
    - `r` rejects the predicted mask. (Reject button)
    - `a` and `d` to cycle through images in your your set. (Next and Prev)
    - `l` and `k` to increase and decrease the transparency of the other annotations.
    - `Ctrl + S` to save progress to the COCO-style JSON file.
7. [coco-viewer](https://github.com/trsvchn/coco-viewer) to view your annotations.
    - `python cocoviewer.py -i <dataset> -a <dataset>/annotations.json`

## Demo

![How it Works Gif!](https://github.com/anuragxel/salt/raw/main/assets/how-it-works.gif)

## Contributing

Follow these guidelines to ensure that your contributions can be reviewed and merged. Need a lot of help in making the UI better.

If you have found a bug or have an idea for an improvement or new feature, please create an issue on GitHub. Before creating a new issue, please search existing issues to see if your issue has already been reported. 

When creating an issue, please include as much detail as possible, including steps to reproduce the issue if applicable.

Create a pull request (PR) to the original repository. Please use `black` formatter when making code changes.

## License

The code is licensed under the MIT License. By contributing to SALT, you agree to license your contributions under the same license as the project. See LICENSE for more information.
