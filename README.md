## Usage
### Grid Search
In order to enable efficient model (hyper)parameter exploration,
this package offers grid search that allows to run a defined number of models
for specified configurations. To specify configurations, 
a JSON file is used (see below for further explanations).
To run grid search, run the python script ``grid_search.py`` via 
``python grid_search.py`` or use the cli entry point ``trans-grid-search``. 

The following parameters are available:
* ``--config`` path to the JSON config file (required)
* ``--ouput`` path to the output directory (required)
* ``--parallel-jobs`` number of jobs (i.e., trainings) that are run in parallel (on CPU and GPU)
* ``--ensemble`` bool indicating whether to produce ensemble results or not

The command ``trans-grid-search --help`` can be run to get information about 
the available parameters.

#### Configuration file
The JSON-based configuration file needs to be passed via ``--config`` parameter.
It basically contains information about the used data as well as model (hyper)parameters.
An example can be found in the test_data folder. The schema for the JSON file is
defined as following:

SCHEMA

Additionally, a simple example can be found INSERT LINK ON GITHUB.

In principle, all parameter values can either be passed as a single value or 
as an array of values. In any case, all possible combinations of all passed
parameter values for a specific grid will be produced and used for training. However,
two things should be noted:
* Firstly, for parameters without required values (e.g., ``--nfd``) a boolean needs
to be specified.
* Secondly, the model parameter ``--sed-params`` expects a dictionary that contains
key value pairs of language name and path to the sed parameters. If a key for a
specific language is missing, a new sed aligner will be trained.

#### Output structure
All output will be generated in the folder specified by the ``--output`` cli argument.
This folder contains a separate folder for each grid that is specified in the config folder.
The name is defined by the name used as key values in the ``grids`` property 
of the config file. This folder contains a `combinations.json` file that 
describes the different possible combinations and maps each combination to a number.
Additionally, this folder contains a separate folder for each trained language.
Each of these "language folders" contains a folder for each possible grid combination
(--> number from `combinations.json`) which, in turn, contain all the trained
models for this specific configuration. Additionally, a results text file is produced
that documents the performance average (accuracy) of all runs. If the ``--ensemble``
parameter is passed, separate results text files will be produced.