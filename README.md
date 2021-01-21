# Neural transducer baseline

This package contains a g2p neural transducer baseline, similar to that used by Makarov & Clematide 2020.

Please make sure that you are using Python3.7.
To run this in a Python virtual environment, perform the following steps:

* Clone this branch and change to the package directory:

        git clone --single-branch -b feature/sgm2021 https://github.com/peter-makarov/il-reimplementation.git neural_transducer
        cd neural_transducer

* Create and activate an environment:

        python3.7 -m venv venv
        source venv/bin/activate

* Install the requirements:

        pip install -U pip
        pip install -r requirements.txt

* Optionally, run unit tests:

        python setup.py test

* Train the models and decode. This will take a while.

        ./sweep

Results will appear in the `output` directory.