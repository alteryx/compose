from setuptools import find_packages, setup

extras_require = {
    'update_checker': ['alteryx-open-src-update-checker >= 2.0.0'],
}

setup(
    name='composeml',
    version='0.7.0',
    author='Feature Labs, Inc.',
    author_email='support@featurelabs.com',
    license='BSD 3-clause',
    url='https://compose.alteryx.com',
    install_requires=open('requirements.txt').readlines(),
    tests_require=open('test-requirements.txt').readlines(),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.6',
    include_package_data=True,
    extras_require=extras_require,
)
