from setuptools import setup


setup(name="neural_transducer",
      version="0.2",
      description=("Neural transducer for grapheme-to-phoneme "
                   "(Makarov & Clematide 2020)"),
      author="Peter Makarov & Simon Clematide",
      author_email="makarov@cl.uzh.ch",
      license="Apache License 2.0",
      packages=["trans"],
      test_suite="trans",
      install_requires=[
          "torch==1.10.1",
          "editdistance>=0.5.2",
          "numpy==1.19.2",
          "progressbar>=2.5",
          "scipy>=1.5.4",
      ],
      python_requires="==3.7.*",
      )
