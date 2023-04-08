# Segment Anything Labelling Tool (SALT)

Uses the Segment-Anything Model By Meta AI and adds a barebones interface to label images.

Under active development, apologies for rough edges and bugs. Use at your own risk.

## Installation

1. Install [Segment Anything](https://github.com/facebookresearch/segment-anything) on any machine with a GPU. (Need not be the labelling machine.)
2. Create a conda environment using `conda conda env create environment.yaml`

## Usage

1. Setup your dataset in the following format `<dataset_name>/images/*`
2. Call `extract_embeddings.py` provided in `helpers/` to extract embeddings for your images.
3. Call `generate_onnx.py` provided in `helpers/` to generate `*.onnx` files in models.
4. Copy the models in `models` folder. Symlink your dataset in the root folder as `<dataset_name>`
4. Call `cocoeditor.py` with argument `<dataset_name>` and categories as `cat1,cat2,cat3..`. 

## Demo

![Alt Text](https://github.com/anuragxel/salt/blob/master/assets/how-it-works.gif)

## Contributing

We welcome contributions to SALT! Please follow these guidelines to ensure that your contributions can be reviewed and merged.

If you have found a bug or have an idea for an improvement or new feature, please create an issue on GitHub. Before creating a new issue, please search existing issues to see if your issue has already been reported.

When creating an issue, please include as much detail as possible, including steps to reproduce the issue if applicable.

Create a pull request (PR) to the original repository. Please use black linter when making code changes.



## License

MIT License. See LICENSE for more information.