from setuptools import setup
from distutils.command.build import build
import subprocess

class Build(build):
    def run(self):
        # Run the original build command
        build.run(self)
        # Custom build stuff goes here
        protoc_command = ["make", "trans"]
        if subprocess.call(protoc_command) != 0:
            exit(-1)
        build.run(self)

setup(name='neural_transducer',
      version='0.1',
      description=('python3 version of neural transducer trained with imitation learning ' 
                   '(Makarov & Clematide EMNLP 2018)'),
      url='',
      author='Peter Makarov & Simon Clematide',
      author_email='makarov@cl.uzh.ch',
      license='MIT',
      packages=['trans'],
      cmdclass={
            'build': Build
      },
      zip_safe=False)
