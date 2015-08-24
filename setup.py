from setuptools import setup, find_packages

setup(
    name='pyltr',
    version='0.1.0',
    description='Python learning to rank (LTR) toolkit.',
    author='Jerry Ma',
    author_email='jerryma@nospam',
    license='BSD-new',

    packages=find_packages(),
    package_data={
        '': ['*.txt', '*.rst'],
    },

    install_requires=[
        'numpy',
        'overrides',
        'scipy',
        'sklearn',
    ],
)
