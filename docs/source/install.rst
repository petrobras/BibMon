Installation
========================

This section covers the installation of BibMon.

Installing from PyPI
--------------------------------

To install BibMon, you can use pip::

    $ pip install bibmon

Installing from conda-forge
--------------------------------

Alternatively, you can use conda::

    $ conda install conda-forge::bibmon

Get the Source Code
-------------------

BibMon is developed on GitHub, where the code is available `here <https://github.com/petrobras/bibmon>`_.

You can either clone the public repository::

    $ git clone https://github.com/petrobras/bibmon.git

Or, download the `tarball <https://github.com/petrobras/bibmon/tarball/main>`_::

    $ curl -OL https://github.com/petrobras/bibmon/tarball/main
    # optionally, zipball is also available (for Windows users).

Once you have a copy of the source, you can embed it in your own Python
package, or install it into your site-packages easily::

    $ cd bibmon
    $ python -m pip install .