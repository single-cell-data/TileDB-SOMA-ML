==============
TileDB-SOMA-ML
==============

|TileDB-SOMA-ML| is a machine learning library for working with |SOMA| data formats.

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

``common``
^^^^^^^^^^

.. automodule:: tiledbsoma_ml.common
   :members:
   :undoc-members:

Batching and Data Management
----------------------------

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

``_distributed``
^^^^^^^^^^^^^^^^

.. automodule:: tiledbsoma_ml._distributed
   :members:
   :undoc-members:

CSR
^^^

.. automodule:: tiledbsoma_ml._csr
   :members:
   :undoc-members:
