# TileDB-SOMA-ML
A Python package containing ML tools for use with [TileDB-SOMA].

[![tiledbsoma-ml package on PyPI](https://img.shields.io/pypi/v/tiledbsoma-ml?label=tiledbsoma-ml)][pypi]


Docs: [single-cell-data.github.io/TileDB-SOMA-ML].

**NOTE:** this is a _pre-release_ package, and may be subject to breaking API changes prior to first release.

## Description

The package contains a prototype PyTorch [`IterableDataset`], [`ExperimentDataset`], for use with the [`torch.utils.data.DataLoader`] API.

[notebooks/](notebooks) contains tutorials and examples that use this repo to train toy models. For a general introduction to PyTorch data loading, [see this tutorial][torch data tutorial]. Additional information on the DataLoader/Dataset pattern [can be found here][`torch.data`].

Defects and feature requests should be filed as [a GitHub issue][/issues] in this repo. Please include a reproducible test case in all bug reports.

## Getting Started

### Installing

Install [from PyPI][pypi]:
```bash
pip install tiledbsoma-ml
```

Developers may install editable, from source, in the usual manner -- clone the repo and execute:

```bash
pip install -e .
```

### Documentation

Documentation can be found at [single-cell-data.github.io/TileDB-SOMA-ML], and in the [notebooks] directory.

## Builds

This is a pure Python package. To build a wheel, ensure you have the `build` package installed, and then:

```bash
python -m build .
```

## Version History

See the [CHANGELOG.md] file.

## License

This project is licensed under the MIT License.

## Acknowledgements

The SOMA team is grateful to the Chan Zuckerberg Initiative Foundation [CELLxGENE Census](https://cellxgene.cziscience.com) team for their initial contribution.

[TileDB-SOMA]: https://github.com/single-cell-data/TileDB-SOMA
[`IterableDataset`]: https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
[`ExperimentDataset`]: src/tiledbsoma_ml/dataset.py
[`torch.utils.data.DataLoader`]: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
[/issues]: https://github.com/single-cell-data/TileDB-SOMA-ML/issues/new
[pypi]: https://pypi.org/project/tiledbsoma-ml/
[torch data tutorial]: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
[`torch.data`]: https://pytorch.org/docs/stable/data.html
[single-cell-data.github.io/TileDB-SOMA-ML]: https://single-cell-data.github.io/TileDB-SOMA-ML/
[notebooks]: notebooks
[CHANGELOG.md]: CHANGELOG.md
[www]: https://single-cell-data.github.io/TileDB-SOMA-ML/
[single-cell-data.github.io/TileDB-SOMA-ML]: https://single-cell-data.github.io/TileDB-SOMA-ML/
