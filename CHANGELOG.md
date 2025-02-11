
# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog] and this project adheres to [Semantic Versioning].

## [Unreleased] - yyyy-mm-dd

Initial release of a [PyTorch Dataset] for [SOMA] Experiments. This is a port and enhancement of code contributed by the Chan Zuckerberg Initiative Foundation [CELLxGENE] project.

This is not a one-for-one migration of the contributed code. Substantial changes have been made to the package utility (e.g., multi-GPU support), improved API UX, performance improvements, and more.

### Added

#### 2025-02
- [#26]:
  - Removed `ExperimentAxisQueryIterDataPipe`, renamed `ExperimentAxisQueryIterableDataset` to `ExperimentDataset`
  - Add `docs`/`www` builds
  - Rename `X_name` kwarg to `layer_name`

#### 2024-12
- [#23]: Split `{test_,}pytorch.py` into a few files
- [#20]: Add `vfs.s3.no_sign_request` config
- [#19]: Support `somacore>=1.0.24` / `tiledbsoma>=1.15`

#### 2024-10
- [#13]: I/O buffer performance optimization
- [#11]: Archive script used to populate the repo commit history
- [#10]: Add first draft of tutorial notebooks
- [#9]: Add shuffling support
- [#8]: Add a DataLoader creation wrapper function
- [#7]: Add CI workflows
- [#6]: Simple, non-shuffling Dataset/DataPipe implementation
- [#4]: Initial project organization and other scaffolding

### Changed

### Fixed


[Keep a Changelog]: http://keepachangelog.com/
[Semantic Versioning]: http://semver.org/

[PyTorch Dataset]: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
[SOMA]: https://github.com/single-cell-data/SOMA
[CELLxGENE]: https://cellxgene.cziscience.com/

[#26]: https://github.com/single-cell-data/TileDB-SOMA-ML/pull/26
[#23]: https://github.com/single-cell-data/TileDB-SOMA-ML/pull/23
[#20]: https://github.com/single-cell-data/TileDB-SOMA-ML/pull/20
[#19]: https://github.com/single-cell-data/TileDB-SOMA-ML/pull/19
[#13]: https://github.com/single-cell-data/TileDB-SOMA-ML/pull/13
[#11]: https://github.com/single-cell-data/TileDB-SOMA-ML/pull/11
[#10]: https://github.com/single-cell-data/TileDB-SOMA-ML/pull/10
[#9]: https://github.com/single-cell-data/TileDB-SOMA-ML/pull/9
[#8]: https://github.com/single-cell-data/TileDB-SOMA-ML/pull/8
[#7]: https://github.com/single-cell-data/TileDB-SOMA-ML/pull/7
[#6]: https://github.com/single-cell-data/TileDB-SOMA-ML/pull/6
[#4]: https://github.com/single-cell-data/TileDB-SOMA-ML/pull/4
