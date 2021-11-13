import setuptools
import subprocess
import logging
import torch

def system(command: str):
    output = subprocess.check_output(command, shell=True)
    logging.info(output)

#system(f'pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.10.0+102.html')
#system(f'pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.10.0+102.html')
#system(f'pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.10.0+102.html')
#system(f'pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.10.0+102.html')
#system(f'pip install torch-geometric')


with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name="gpn",
    version="latest",

    author='Maximilian Stadler',
    author_email='maximilian.stadler@tum.de',
    url='https://github.com/anonymous/anonymous',

    description="Graph Posterior Network: Bayesian Predictive Uncertainty for Node Classification",

    long_description=long_description,
    long_description_content_type="text/markdown",

    packages=setuptools.find_packages(),

    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    license='License :: OSI Approved :: MIT License',
    zip_safe=False
)