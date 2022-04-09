from setuptools import setup
import sys

install_requires = [
          "Cython",
          "torch==1.10.1",
          "editdistance>=0.5.2",
          "numpy==1.19.2",
          "progressbar>=2.5",
          "scipy>=1.5.4",
]

if "--cuda" in sys.argv[1:]:
    install_requires.append("cudatoolkit==11.3")
    sys.argv.remove("--cuda")
    print(install_requires)


setup(name="neural_transducer",
      version="0.2",
      description=("Neural transducer for grapheme-to-phoneme "
                   "(Makarov & Clematide 2020)"),
      author="Peter Makarov, Simon Clematide & Silvan Wehrli",
      author_email="makarov@cl.uzh.ch",
      license="Apache License 2.0",
      packages=["trans"],
      test_suite="trans.tests",
      install_requires=install_requires,
      python_requires="==3.7.*",
      entry_points={
          "console_scripts": [
              "trans-train = trans.train:cli_main",
              "trans-ensemble = trans.ensembling:cli_main",
              "trans-grid-search = trans.grid_search:cli_main"
          ]
      },
      )
