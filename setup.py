from setuptools import setup, find_packages

setup(
    name='pyltr',
    version='0.2.0',
    description='Python learning to rank (LTR) toolkit.',
    author='Jerry Ma',
    author_email='jerryma_nospam@example.org',
    url='https://github.com/jma127/pyltr',
    license='BSD-new',

    packages=find_packages(),
    package_data={
        '': ['*.txt', '*.rst'],
    },

    setup_requires=[
        'scipy',
    ],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
    ],

    zip_safe=True,
)
