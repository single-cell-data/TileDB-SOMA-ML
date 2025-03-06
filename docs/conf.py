import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "TileDB-SOMA-ML"
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


def setup(app):
    # Add '_blank' target to all external links
    app.add_js_file("js/external_links.js")
    app.add_css_file("custom.css")


extensions = [
    "sphinx.ext.autodoc",  # Auto-generate docs from docstrings
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",  # Support Google/NumPy docstring style
    "sphinx.ext.viewcode",  # Add links to source code
    "sphinx.ext.intersphinx",  # Link to other projects' documentation
]

mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
    }
}

# Link to external Python docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "tiledbsoma": ("https://tiledbsoma.readthedocs.io/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

doctest_test_doctest_blocks = None  # Don't try to run any code samples

nitpicky = True
nitpick_ignore = {
    # These lack usable external link targets
    ("py:class", "numpy.int64"),
    ("py:class", "numpy._typing._dtype_like._SupportsDType"),
    ("py:class", "numpy._typing._dtype_like._DTypeDict"),
}

python_use_unqualified_type_names = True  # Enables shorter references

rst_prolog = """
.. |List| replace:: :class:`~typing.List`
.. |Iterable| replace:: :class:`~typing.Iterable`
.. |Iterator| replace:: :class:`~typing.Iterator`
.. |ExperimentDataset| replace:: :class:`~tiledbsoma_ml.dataset.ExperimentDataset`
.. |Encoder| replace:: :class:`~tiledbsoma_ml.encoders.Encoder`
.. |__iter__| replace:: :obj:`__iter__ <.__iter__>`
.. |XLocator| replace:: :class:`~tiledbsoma_ml.x_locator.XLocator`
.. |ExperimentDataset.__iter__| replace:: :obj:`ExperimentDataset.__iter__ <tiledbsoma_ml.ExperimentDataset.__iter__>`
.. |batch_size| replace:: :attr:`batch_size <tiledbsoma_ml.ExperimentDataset.batch_size>`
.. |io_batch_size| replace:: :attr:`io_batch_size <tiledbsoma_ml.ExperimentDataset.io_batch_size>`
.. |shuffle_chunk_size| replace:: :attr:`shuffle_chunk_size <tiledbsoma_ml.ExperimentDataset.shuffle_chunk_size>`
.. |return_sparse_X| replace:: :attr:`return_sparse_X <tiledbsoma_ml.ExperimentDataset.return_sparse_X>`
.. |experiment_dataloader| replace:: :obj:`~tiledbsoma_ml.experiment_dataloader`
.. |MiniBatch| replace:: :class:`~tiledbsoma_ml._common.MiniBatch`
.. |mini batches| replace:: :class:`"mini batches" <tiledbsoma_ml._common.MiniBatch>`
.. |NDArrayJoinId| replace:: :class:`~tiledbsoma_ml._common.NDArrayJoinId`
.. |Partition| replace:: :class:`~tiledbsoma_ml._query_ids.Partition`
.. |QueryIDs| replace:: :class:`~tiledbsoma_ml._query_ids.QueryIDs`
.. |obs_joinids| replace:: :obj:`~tiledbsoma_ml._query_ids.QueryIDs.obs_joinids`
.. |QueryIDs.shuffle_chunks| replace:: :meth:`QueryIDs.shuffle_chunks <tiledbsoma_ml._query_ids.QueryIDs.shuffle_chunks>`
.. |QueryIDs.random_split| replace:: :meth:`QueryIDs.random_split <tiledbsoma_ml._query_ids.QueryIDs.random_split>`
.. |Chunks| replace:: :class:`~tiledbsoma_ml._query_ids.Chunks`
.. |SamplingMethod| replace:: :class:`~tiledbsoma_ml._query_ids.SamplingMethod`
.. |IOBatch| replace:: :class:`~tiledbsoma_ml._io_batch_iterable.IOBatch`
.. |IOBatchIterable| replace:: :class:`~tiledbsoma_ml._io_batch_iterable.IOBatchIterable`
.. |CSR_IO_Buffer| replace:: :class:`~tiledbsoma_ml._csr.CSR_IO_Buffer`
.. |get_distributed_rank_and_world_size| replace:: :obj:`~tiledbsoma_ml._distributed.get_distributed_rank_and_world_size`
.. |get_worker_id_and_num| replace:: :obj:`~tiledbsoma_ml._distributed.get_worker_id_and_num`
.. |TileDB-SOMA-ML| replace:: TileDB-SOMA-ML_
.. _TileDB-SOMA-ML: https://github.com/single-cell-data/TileDB-SOMA-ML
.. |SOMA| replace:: SOMA_
.. _SOMA: https://github.com/single-cell-data/SOMA
.. |DataFrame| replace:: :class:`~tiledbsoma.DataFrame`
.. |Experiment| replace:: :class:`~tiledbsoma.Experiment`
.. |ExperimentAxisQuery| replace:: :class:`~tiledbsoma.ExperimentAxisQuery`
.. |Measurement| replace:: :class:`~tiledbsoma.Measurement`
.. |SparseNDArray| replace:: :class:`~tiledbsoma.SparseNDArray`
.. |torch.distributed| replace:: :mod:`torch.distributed`
.. |torch.Tensor| replace:: :class:`~torch.Tensor`
.. |DataLoader| replace:: :class:`~torch.utils.data.DataLoader`
.. |IterableDataset| replace:: :class:`~torch.utils.data.IterableDataset`
.. |DistributedDataParallel| replace:: :class:`~torch.nn.parallel.DistributedDataParallel`
.. |ndarray| replace:: :class:`~numpy.ndarray`
.. |np.ndarray| replace:: :class:`np.ndarray <numpy.ndarray>`
.. |pd.DataFrame| replace:: :class:`pd.DataFrame <pandas.DataFrame>`
.. |csr_matrix| replace:: :class:`~scipy.sparse.csr_matrix`
"""
default_domain = "py"
