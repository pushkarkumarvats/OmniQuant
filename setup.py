"""
OmniQuant - Unified Quantitative Research & Trading Framework
Setup configuration
"""

from setuptools import setup, find_packages, Extension
from pathlib import Path
import sys

# Read README
long_description = (Path(__file__).parent / "README.md").read_text()

# C++ Extension for high-performance simulator (optional)
# Only build if pybind11 is available AND C++ source files exist
ext_modules = []
cmdclass = {}
try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    
    cpp_sources = [
        "src/simulator/core/orderbook.cpp",
        "src/simulator/core/matching_engine.cpp",
        "src/simulator/core/simulator.cpp",
        "src/simulator/bindings.cpp",
    ]
    if all(Path(f).exists() for f in cpp_sources):
        ext_modules = [
            Pybind11Extension(
                "omniquant.simulator_core",
                cpp_sources,
                include_dirs=["src/simulator/include"],
                cxx_std=17,
                extra_compile_args=["-O3", "-march=native"] if sys.platform != "win32" 
                                  else ["/O2"],
            ),
        ]
        cmdclass = {"build_ext": build_ext}
    else:
        print("Note: C++ source files not found. Skipping native extensions.")
except ImportError:
    print("Note: pybind11 not found. C++ extensions will not be built.")

setup(
    name="omniquant",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Unified Quantitative Research & Trading Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/omniquant",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.1.0",
        "cvxpy>=1.4.0",
        "plotly>=5.17.0",
        "streamlit>=1.27.0",
        "pydantic>=2.4.0",
        "loguru>=0.7.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "ml": [
            "stable-baselines3>=2.1.0",
            "ray[rllib]>=2.7.0",
            "transformers>=4.30.0",
        ],
        "causal": [
            "dowhy>=0.11.0",
            "econml>=0.14.0",
        ],
        "full": [
            # All optional dependencies
        ],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    entry_points={
        "console_scripts": [
            "omniquant=omniquant.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
