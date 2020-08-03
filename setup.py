import os
from setuptools import setup, Extension
from distutils import sysconfig
from distutils.command.build_ext import build_ext

libalign = Extension('trans.libalign', sources=['trans/align.c'])


class NoSuffixBuilder(build_ext):
    def get_ext_filename(self, ext_name):
        filename = super().get_ext_filename(ext_name)
        suffix = sysconfig.get_config_var('EXT_SUFFIX')
        ext = os.path.splitext(filename)[1]
        return filename.replace(suffix, "") + ext


setup(name='neural_transducer',
      version='0.1',
      description=('python3 version of neural transducer trained with imitation learning '
                   '(Makarov & Clematide EMNLP 2018)'),
      author='Peter Makarov & Simon Clematide',
      author_email='makarov@cl.uzh.ch',
      license='MIT',
      packages=['trans'],
      ext_modules=[libalign],
      cmdclass={
        "build_ext": NoSuffixBuilder,
      },
      install_requires=[
        "wheel==0.34.2",
        "Cython==0.29",
        "docopt==0.6.2",
        "dyNET==2.1",
        "editdistance==0.5.2",
        "numpy==1.15.4",
        "progressbar==2.5",
      ])
