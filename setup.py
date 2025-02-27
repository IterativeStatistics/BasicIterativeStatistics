from setuptools import setup, find_packages

setup(
    name="iterative_stats",
    version="0.1.1",
    description="This package implements iterative algorithms to compute some basics statistics",
    author="Frederique Robin",
    license="BSD License",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9,<3.13",
    install_requires=[
        "pyyaml>=6.0",
        "numpy>=1.19.0,<2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.1",
            "autopep8==1.6.0",
            "openturns>=1.19,<2.0.0",
            "scipy>=1.8.0,<2.0.0",
            "mypy>=1.4.0,<2.0.0",
            "bump2version>1.0.0"
        ]
    },
    packages=find_packages(),
    test_suite="tests",
    entry_points={
        "console_scripts": [
            # Add any console scripts here if applicable
        ],
    },
    url="https://github.com/IterativeStatistics/BasicIterativeStatistics",
    project_urls={
        "Homepage": "https://github.com/IterativeStatistics/BasicIterativeStatistics",
        "Bug Tracker": "https://github.com/IterativeStatistics/BasicIterativeStatistics/issues",
    },
)
