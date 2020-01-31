from setuptools import find_packages, setup

setup(
    name='composeml',
    version='0.1.7',
    author='Feature Labs, Inc.',
    author_email='support@featurelabs.com',
    license='BSD 3-clause',
    url='https://compose.featurelabs.com',
    install_requires=open('requirements.txt').readlines(),
    tests_require=open('test-requirements.txt').readlines(),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    package_data={'composeml': ['demos/*.csv']},
    include_package_data=True,
)
