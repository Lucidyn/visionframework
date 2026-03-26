from setuptools import setup, find_packages
from pathlib import Path

readme = Path(__file__).parent / "README.md"
long_description = readme.read_text(encoding="utf-8") if readme.exists() else ""

setup(
    name="visionframework",
    version="2.0.0",
    description="Modular, component-based computer vision framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["test", "test.*"]),
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
        "rtdetr-verify": ["ultralytics>=8.4.0"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "vf-test-yolo26=visionframework.tools.test_yolo26:main",
            "vf-convert-ultralytics=visionframework.tools.convert_ultralytics:main",
            "vf-convert-detr=visionframework.tools.convert_detr:main",
            "vf-convert-rtdetr=visionframework.tools.convert_ultralytics_rtdetr_hg:main",
            "vf-run=visionframework.tools.run_inference:main",
        ]
    },
)
