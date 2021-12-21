from setuptools import setup

setup(
    name = 'DRLagents',
    version= '0.0.1',
    description= 'A toolkit to train DeepRL-agents',
    author= 'Dibyojyoti Sinha',
    author_email= 'dibyo@iitk.ac.in',
    packages= ['DRLagents'],
    py_modules=['agents', 'explorationStrategies', 'replaybuffers', 'utils'],
    install_requires=['numpy', 'gym', 'torch']
)