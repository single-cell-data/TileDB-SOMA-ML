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

``query_ids``
^^^^^^^^^^^^^

.. automodule:: tiledbsoma_ml.query_ids

QueryIDs
~~~~~~~~
.. autoclass:: QueryIDs
   :members:
   :undoc-members:
   :member-order: bysource

Partition
~~~~~~~~~
.. autoclass:: Partition
   :members:
   :undoc-members:
   :member-order: bysource

Chunks
~~~~~~
.. autoclass:: Chunks

SamplingMethod
~~~~~~~~~~~~~~
.. autoclass:: SamplingMethod

``io_batches``
^^^^^^^^^^^^^^

.. automodule:: tiledbsoma_ml.io_batches
   :members:
   :undoc-members:

``mini_batch_iterable``
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tiledbsoma_ml.mini_batch_iterable
   :members:
   :undoc-members:

``x_locator``
^^^^^^^^^^^^^

.. automodule:: tiledbsoma_ml.x_locator
   :members:

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
