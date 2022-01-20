from os import path

from setuptools import find_packages, setup

dirname = path.abspath(path.dirname(__file__))
with open(path.join(dirname, 'README.md')) as f:
    long_description = f.read()


extras_require = {
    'update_checker': ['alteryx-open-src-update-checker >= 2.0.0'],
}

setup(
    name='composeml',
    version='0.8.0',
    author='Feature Labs, Inc.',
    author_email='support@featurelabs.com',
    license='BSD 3-clause',
    url='https://compose.alteryx.com',
    classifiers=[
         'Development Status :: 3 - Alpha',
         'Intended Audience :: Developers',
         'Programming Language :: Python :: 3',
         'Programming Language :: Python :: 3.7',
         'Programming Language :: Python :: 3.8',
         'Programming Language :: Python :: 3.9',
    ],
    packages=find_packages(),
    install_requires=open('requirements.txt').readlines(),
    tests_require=open('test-requirements.txt').readlines(),
    python_requires='>=3.7, <4',
    extras_require=extras_require,
    keywords='data science machine learning',
    include_package_data=True,
    long_description=long_description,
    long_description_content_type='text/markdown'
)
