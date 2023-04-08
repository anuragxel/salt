# Segment Anything Labelling Tool (SALT)

Uses the Segment Anything Tool By Meta AI and builds a super barebones interface to label images using the tool.

Under active development, so kindly apologize for rough edged and any bugs. Use at your own risk.

## Installation

1. Install [Segment Anything](https://github.com/facebookresearch/segment-anything) on any machine with a GPU.


## Usage

1. Setup your dataset in the following format `<dataset_name>/images/*`
2. Call `extract_embeddings.py` provided in `helpers/` to extract embeddings for your images.
3. Call `generate_onnx.py` provided in `helpers/` to generate `*.onnx` files in models.
4. Call `cocoeditor.py` with argument `<dataset_name>`. 

## Examples

## Contributing

We welcome contributions to SALT! Please follow these guidelines to ensure that your contributions can be reviewed and merged.

If you have found a bug or have an idea for an improvement or new feature, please create an issue on GitHub. Before creating a new issue, please search existing issues to see if your issue has already been reported.

When creating an issue, please include as much detail as possible, including steps to reproduce the issue if applicable.

Create a pull request (PR) to the original repository. Please use black linter when making code changes.



## License

MIT License. See LICENSE for more information.