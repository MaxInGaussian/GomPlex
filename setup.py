################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = '0.1.0'

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

setup(
    name='GPoFM',
    version=__version__,
    description='GomPlex: Complex Gaussian Process for Machine Learning',
    long_description=long_description,
    url='https://github.com/MaxInGaussian/GomPlex',
    download_url='https://github.com/MaxInGaussian/GomPlex/tarball/' + __version__,
    license='BSD',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Programming Language :: Python :: 3',
    ],
    keywords='gaussian process, complex number, machine learning, regression',
    packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True,
    author='Max W. Y. Lam',
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email='maxingaussian@gmail.com'
)
