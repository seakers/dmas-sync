from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='dmas',
    version='2.0.0',
    description='Decentralized, Distributed, Dynamic, and Context-aware Heterogeneous Sensor Systems',
    author='SEAK Lab',
    author_email='aguilaraj15@tamu.edu',
    packages=['dmas'],
    scripts=[],
    install_requires=['tqdm', 'instrupy', 'orbitpy', 'execsatm', 'networkx',
                       'pytest', 'gurobipy', 'pandas', 'numpy', 'pyarrow', 'fastparquet'],
    long_description=readme(),
)
