==============
TileDB-SOMA-ML
==============

|TileDB-SOMA-ML| is a machine learning library for working with |SOMA| data formats.

See `/shuffle <./shuffle>`_ for a visualization of the batching/shuffling algorithm:

.. figure:: ./shuffle/shuffle.gif
   :alt: Animation showing the stages of batching and shuffling TileDB-SOMA-ML performs
   :target: ./shuffle

.. contents:: Table of Contents
   :depth: 2
   :local:

Module Contents
---------------

``dataset``
^^^^^^^^^^^

.. autoclass:: tiledbsoma_ml.ExperimentDataset
   :members:
   :undoc-members:
   :special-members: __init__, __iter__, __len__

``dataloader``
^^^^^^^^^^^^^^

.. autofunction:: tiledbsoma_ml.experiment_dataloader

Batching and Data Management
----------------------------

``common``
^^^^^^^^^^

.. automodule:: tiledbsoma_ml.common
   :members:
   :undoc-members:

``_distributed``
^^^^^^^^^^^^^^^^

.. automodule:: tiledbsoma_ml._distributed
   :members:
   :undoc-members:

``_csr``
^^^^^^^^

.. automodule:: tiledbsoma_ml._csr
   :members:
   :member-order: bysource
   :special-members: __init__
