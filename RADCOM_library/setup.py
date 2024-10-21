from setuptools import find_packages, setup
setup(
    name='radcomlib',
    version='1.0.0',
    packages=['radcomlib'],
	license='LICENSE.txt',
    description='RADCOM Library : useful functions for RADCOM systems implementation.',
	long_description=open('README.txt').read(),
    author='François De Saint Moulin',
	install_requires=['numpy','scipy','scikit-commpy','PyQt5','numba'],
)