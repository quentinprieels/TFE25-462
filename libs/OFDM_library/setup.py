from setuptools import setup

setup(
    name="ofdmlib",
    version="1.0.0",
    packages=['ofdmlib'],
    license='LICENSE.txt',
    description='OFDM Library: useful functions for OFDM systems implementation.',
    author='Quentin Prieels',
    install_requires=['numpy', 'matplotlib', 'seaborn', 'radcomlib', 'tqdm'],
)