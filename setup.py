from setuptools import setup, find_packages

setup(
    name='gpu_nds',
    version='1.0.0',
    description='GPU-Native Divide-and-Conquer Non-Dominated Sorting',
    author='Ayaan',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'scikit-learn>=1.2.0',
        'matplotlib>=3.7.0',
        'pandas>=2.0.0',
        'tqdm>=4.65.0',
    ],
    extras_require={
        'gpu': ['cupy-cuda11x>=12.0.0'],
    },
)
