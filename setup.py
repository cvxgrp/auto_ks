from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='auto_ks',
    version='0.1',
    url='https://www.github.com/cvxgrp/auto_ks',
    author='Shane Barratt and Stephen Boyd',
    author_email='sbarratt@stanford.edu',
    description=('Implementation of "Fitting a Kalman Smoother to Data".'),
    long_description=readme(),
    license='Apache 2.0',
    packages=['auto_ks'],
    include_package_data=True,
    install_requires=['numpy >= 1.17', 'scipy >= 1.3'],
)
