import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

html_theme = "sphinx_rtd_theme"

# docs/conf.py additions
extensions = [
    "sphinx.ext.autodoc",  # Auto-generate docs from docstrings
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",  # Support Google/NumPy docstring style
    "sphinx.ext.viewcode",  # Add links to source code
    "sphinx.ext.intersphinx",  # Link to other projects' documentation
]

# If you want to link to external Python docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "tiledbsoma": ("https://tiledbsoma.readthedocs.io/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

doctest_test_doctest_blocks = None  # Don't try to run any code samples

nitpicky = True
nitpick_ignore = {
    # These lack usable link targets
    ("py:class", "numpy.int64"),
    ("py:class", "numpy._typing._dtype_like._SupportsDType"),
    ("py:class", "numpy._typing._dtype_like._DTypeDict"),
}

python_use_unqualified_type_names = True  # Enables shorter references

rst_prolog = """
.. |List| replace:: :class:`~typing.List`
.. |Iterable| replace:: :class:`~typing.Iterable`
.. |ExperimentDataset| replace:: :class:`~tiledbsoma_ml.dataset.ExperimentDataset`
.. |ExperimentDataset.__iter__| replace:: :obj:`ExperimentDataset.__iter__ <tiledbsoma_ml.ExperimentDataset.__iter__>`
.. |experiment_dataloader| replace:: :func:`~tiledbsoma_ml.experiment_dataloader`
.. |Batch| replace:: :class:`~tiledbsoma_ml.common.Batch`
.. |NDArrayJoinId| replace:: :class:`~tiledbsoma_ml.common.NDArrayJoinId`
.. |CSR_IO_Buffer| replace:: :class:`~tiledbsoma_ml._csr.CSR_IO_Buffer`
.. |Partition| replace:: :class:`~tiledbsoma_ml.query_ids.Partition`
.. |QueryIDs| replace:: :class:`~tiledbsoma_ml.query_ids.QueryIDs`
.. |Chunks| replace:: :class:`~tiledbsoma_ml.query_ids.Chunks`
.. |IOBatch| replace:: :class:`~tiledbsoma_ml.io_batches.IOBatch`
.. |IOBatches| replace:: :class:`~tiledbsoma_ml.io_batches.IOBatches`
.. |get_distributed_world_rank| replace:: :obj:`~tiledbsoma_ml._distributed.get_distributed_world_rank`
.. |get_worker_world_rank| replace:: :obj:`~tiledbsoma_ml._distributed.get_worker_world_rank`
.. |DataFrame| replace:: :class:`~tiledbsoma.DataFrame`
.. |Experiment| replace:: :class:`~tiledbsoma.Experiment`
.. |ExperimentAxisQuery| replace:: :class:`~tiledbsoma.ExperimentAxisQuery`
.. |Measurement| replace:: :class:`~tiledbsoma.Measurement`
.. |SparseNDArray| replace:: :class:`~tiledbsoma.SparseNDArray`
.. |DataLoader| replace:: :class:`~torch.utils.data.DataLoader`
.. |IterableDataset| replace:: :class:`~torch.utils.data.IterableDataset`
.. |DistributedDataParallel| replace:: :class:`~torch.nn.parallel.DistributedDataParallel`
.. |ndarray| replace:: :class:`~numpy.ndarray`
.. |np.ndarray| replace:: :class:`np.ndarray <numpy.ndarray>`
.. |pd.DataFrame| replace:: :class:`pd.DataFrame <pandas.DataFrame>`
.. |csr_matrix| replace:: :class:`~scipy.sparse.csr_matrix`
.. |Iterator| replace:: :class:`~typing.Iterator`
"""
default_domain = "py"
