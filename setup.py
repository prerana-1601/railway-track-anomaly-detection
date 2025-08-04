"""
Setup script for Railway Track Anomaly Detection System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="railway-track-anomaly-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A real-time computer vision system for detecting anomalies in railway tracks using drone-captured video streams and MQTT communication.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/railway-track-anomaly-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "pillow>=8.3.0",
        "pandas>=1.3.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "ultralytics>=8.0.0",
        "PyYAML>=6.0",
        "paho-mqtt>=1.6.0",
        "reportlab>=3.6.0",
        "requests>=2.25.0",
        "python-dateutil>=2.8.0",
        "geopy>=2.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "gpu": [
            "torch>=1.9.0+cu111",
        ],
    },
    entry_points={
        "console_scripts": [
            "railway-drone=client.drone_client:main",
            "railway-detector=server.anomaly_detector:main",
            "railway-mqtt=server.mqtt_server:main",
            "railway-pdf=server.pdf_generator:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
) 