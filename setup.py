import os
import subprocess

from setuptools import find_packages, setup

with open(os.path.join("rl-baselines3-zoo", "version.txt")) as file_handler:
    __version__ = file_handler.read().strip()

# Taken from PyTorch code to have a different version per commit
hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=".").decode("ascii").strip()

setup(
    name="stable_baselines3",
    packages=[package for package in find_packages() if package.startswith("stable_baselines3")],
    package_data={"stable_baselines3": ["py.typed", "version.txt"]},
    install_requires=[
        "gym==0.21",  # Fixed version due to breaking changes in 0.22
        "numpy",
        "torch>=1.11",
        # For saving models
        "cloudpickle",
        # For reading logs
        "pandas",
        # Plotting learning curves
        "matplotlib",
    ],
    extras_require={
        "tests": [
            # Run tests and coverage
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-xdist",
            # Type check
            "pytype",
            # Lint code
            "flake8>=3.8",
            # Find likely bugs
            "flake8-bugbear",
            # Sort imports
            "isort>=5.0",
            # Reformat
            "black",
            # For toy text Gym envs
            "scipy>=1.4.1",
        ],
        "docs": [
            "sphinx",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            # For spelling
            "sphinxcontrib.spelling",
            # Type hints support
            "sphinx-autodoc-typehints",
        ],
        "extra": [
            # For render
            "opencv-python",
            # For atari games,
            "ale-py==0.7.4",
            "autorom[accept-rom-license]~=0.4.2",
            "pillow",
            # Tensorboard support
            "tensorboard>=2.2.0",
            # Checking memory taken by replay buffer
            "psutil",
            # Additional environments
            "my_gym",
        ],
    },
    description="RL zoo",
    author="Antonin Raffin",
    url="https://github.com/osigaud/rl-baselines3-zoo",
    author_email="antonin.raffin@dlr.de",
    keywords="reinforcement-learning-algorithms reinforcement-learning machine-learning "
    "gym openai stable baselines toolbox python data-science",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=f"{__version__}+{hash}",
    python_requires=">=3.7",
    # PyPI package information.
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)

# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
