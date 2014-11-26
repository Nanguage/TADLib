Introduction
------------
Chromosome conformation capture (3C) derived techniques, especially Hi-C,
have revealed that topologically associating domain (TAD) is a structural
basis for both chromatin organization and regulation in three-dimensional
(3D) space. To systematically investigate the relationship between structure
and function, it is important to develop a quantitative parameter to measure
the structural characteristics of TAD. TADLib is such a package to explore
the chromatin interaction pattern of TAD.

Inspired by the observation that there exist great differences in chromatin
interaction pattern and gene expression level among TADs, a chromatin interaction
feature is developed to capture the aggregation degree of long-range chromatin
interactions. Application to human and mouse cell lines shows that there
exist heterogeneous structures among TADs and the structural rearrangement across
cell lines is significantly associated with transcription activity remodeling.

TADLib package is written in Python and provides a four-step pipeline:

- Identifying TAD from Hi-C data (optional)
- Selecting long-range chromatin interactions in each TAD
- Finding the aggregation patterns of selected interactions
- Calculating chromatin interaction feature of TAD

Installation
------------
Please check the file "INSTALL.rst" in the distribution.

Links
-----
- `Detailed Documentation <http://pythonhosted.org//TADLib/>`_
- `Repository <https://github.com/XiaoTaoWang/TADLib>`_ (At GitHub)
- `PyPI <https://pypi.python.org/pypi/TADLib>`_ (Download and Installation)

Notes
-----
Although not required, correction procedures, such as [1]_ or [2]_ are recommended
for original Hi-C data to eliminate systematic biases.

.. [1] Imakaev M, Fudenberg G, McCord RP et al. Iterative correction of Hi-C data
   reveals hallmarks ofchromosome organization. Nat Methods, 2012, 9: 999-1003.

.. [2] Yaffe E, Tanay A. Probabilistic modeling of Hi-C contact maps eliminates
   systematic biases to characterize global chromosomal architecture. Nat Genet,
   2011, 43: 1059-65.
