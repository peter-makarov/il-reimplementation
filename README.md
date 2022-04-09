# A G2P neural transducer
This package contains a cli-based g2p neural transducer, similar to that used by Makarov & Clematide 2020.

## Installation
Please make sure that you are using Python 3.7.
To install this package, perform the following steps:

* Clone the development branch and change to the package directory:

        git clone --single-branch -b development https://github.com/slvnwhrl/il-reimplementation.git neural_transducer
        cd neural_transducer

* Install the package:

  * default installation

        pip install
  
  * with cuda support
        
        pip install --cuda

  * local development (without the need to reinstall the package after changes):

        pip install -e ./

* Optionally, run unit tests:

        python setup.py test

## Usage
### Training
In order to train a model, directly run the python script ``train.py`` 
via ``python train.py`` or use the cli entry point ``trans-train``.

The most important (and required) parameters are:
* ``--train`` path to the training data
* ``--dev`` path to the development data
* ``--output`` path to the output directory

For a full list of available training configurations, use ``trans-train --help``.

### Ensembling
To ensemble a number of models based on majority voting, run the python script 
``ensembling.py`` via ``python ensembling.py`` or use the cli entry point 
``trans-ensemble``. The following parameters are required:
* ``--gold`` path to the gold data
* ``--systems`` path to the systems' data
* ``--output`` path to the output directory