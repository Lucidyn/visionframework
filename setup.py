from setuptools import setup, find_packages
from pathlib import Path

readme = Path(__file__).parent / "README.md"
long_description = readme.read_text(encoding="utf-8") if readme.exists() else ""

setup(
    name="visionframework",
    version="1.0.0",
    description="Modular, component-based computer vision framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["test", "test.*", "configs"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0,<2.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "scipy": ["scipy>=1.10.0"],
        "dev": ["pytest>=7.0", "scipy>=1.10.0"],
    },
)
