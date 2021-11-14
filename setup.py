import setuptools


with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name="gpn",
    version="latest",

    author='Maximilian Stadler',
    author_email='maximilian.stadler@tum.de',
    url='https://github.com/stadlmax/Graph-Posterior-Network',

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
