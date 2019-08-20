from setuptools import find_packages, setup

setup(
    name='composeml',
    version='0.1.4',
    author='Feature Labs, Inc.',
    author_email='support@featurelabs.com',
    license='BSD 3-clause',
    url='http://compose.ml',
    install_requires=open('requirements.txt').readlines(),
    tests_require=open('test-requirements.txt').readlines(),
    packages=find_packages(),
    package_data={'composeml': ['datasets/data/*.csv']},
    include_package_data=True,
)
