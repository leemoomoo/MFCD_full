# setup.py
from setuptools import setup, find_packages

setup(
    name="ncdm",
    version="1.0.0",
    author="易佳程",
    author_email="yij03389@gmail.com",
    description="Hybrid Neural Cognitive Diagnosis Model (Hybrid-NCDM) framework",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ncdm-plus",  # 可改为你的GitHub地址
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.20.0",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.60.0"
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
