from setuptools import setup


setup(name='neural_transducer',
      version='0.1',
      description=('python3 version of neural transducer trained with imitation learning '
                   '(Makarov & Clematide EMNLP 2018)'),
      author='Peter Makarov & Simon Clematide',
      author_email='makarov@cl.uzh.ch',
      license='MIT',
      packages=['trans'],
      install_requires=[
        "wheel==0.34.2",
        "Cython==0.29",
        "docopt==0.6.2",
        "dyNET==2.1",
        "editdistance==0.5.2",
        "numpy==1.15.4",
        "progressbar==2.5",
      ])
