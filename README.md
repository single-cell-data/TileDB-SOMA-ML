
# tiledbsoma_ml

A Python package containing ML tools for use with `tiledbsoma`.

**NOTE:** this is a _pre-release_ package, and may be subject to breaking API changes prior to first release.

## Description

The package contains a prototype PyTorch `IterableDataset` for use with the
[`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
API. For a general introduction to PyTorch data loading,
[see this tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html).
Additional information on the DataLoader/Dataset pattern
[can be found here](https://pytorch.org/docs/stable/data.html).

Defects and feature requests should be filed as a GitHub issue in this repo. Please include a reproducible
test case in all bug reports.

## Getting Started

### Installing

Prior to the first release, installation is most easily accomplished by installing directly from GitHub:

```bash
pip install git+https://github.com/single-cell-data/TileDB-SOMA-ML
```

Developers may install editable, from source, in the usual manner -- clone the repo and execute:

```bash
pip install -e .
```

### Documentation

Documentation is pending. Preliminary documentation can be found in API docstrings, and in
the [notebooks](notebooks) directory.

## Builds

This is a pure Python package. To build a wheel, ensure you have the `build` package installed, and then:

```bash
python -m build .
```

## Version History

See the [CHANGELOG.md](CHANGELOG.md) file.

## License

This project is licensed under the MIT License.

## Acknowledgements

The SOMA team is grateful to the Chan Zuckerberg Initiative Foundation [CELLxGENE Census](https://cellxgene.cziscience.com)
team for their initial contribution.
