from setuptools import find_packages, setup

setup(
    name='composeml',
    version='0.1.0',
    author='Feature Labs, Inc.',
    author_email='support@featurelabs.com',
    install_requires=open('requirements.txt').readlines(),
    tests_require=open('test-requirements.txt').readlines(),
    packages=find_packages(),
)
