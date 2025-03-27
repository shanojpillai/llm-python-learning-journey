from setuptools import setup, find_packages

setup(
    name="text-processing-pipeline",
    version="0.1",
    packages=find_packages(where='.', exclude=['tests*']),
    package_dir={'': '.'},  # This ensures the current directory is used
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'scikit-learn>=1.0.0',
        'torch>=1.9.0',
        'nltk>=3.6.0',
        'transformers>=4.8.0',
        'streamlit>=1.10.0',
        'pytest>=6.2.0'
    ],
)