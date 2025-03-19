# This file is used to package the trading bot as a Python package.
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="goalkickltd_trading_bot",
    version="0.1.0",
    author="Goalkick Ltd",
    author_email="info@goalkickltd.com",
    description="A high-performance cryptocurrency trading bot",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pat435/Goalkickltd",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "goalkick-trading-bot=src.main:main",
        ],
    },
)