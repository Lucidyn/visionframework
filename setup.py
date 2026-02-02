from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="visionframework",
    version="0.2.14",
    description="A comprehensive vision framework for detection, tracking, and more with configuration management, model management, and device autoselection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Vision Framework Contributors",
    author_email="",
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="computer vision, object detection, object tracking, yolo, deep learning, pydantic, configuration management, model management",
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0,<2.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.0",
        "scipy>=1.10.0",
        "Pillow>=10.0.0",
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
        "huggingface_hub>=0.14.0",
    ],
    python_requires=">=3.8",
    extras_require={
        "clip": [
            "transformers>=4.30.0",
        ],
        "detr": [
            "transformers>=4.30.0",
        ],
        "sam": [
            "segment-anything>=1.0",
        ],
        "rfdetr": [
            "rfdetr>=0.1.0",
            "supervision>=0.18.0",
        ],
        "pyav": [
            "av>=11.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "all": [
            "transformers>=4.30.0",
            "segment-anything>=1.0",
            "rfdetr>=0.1.0",
            "supervision>=0.18.0",
            "av>=11.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

