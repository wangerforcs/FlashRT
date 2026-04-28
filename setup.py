"""FlashRT — pip install support.

Usage:
    # Development mode (editable, recommended for development):
    pip install -e .

    # Standard install:
    pip install .

    # With optional dependencies:
    pip install -e ".[torch]"       # PyTorch frontend
    pip install -e ".[jax]"         # JAX frontend
    pip install -e ".[server]"      # FastAPI server
    pip install -e ".[all]"         # Everything

Note: CUDA kernels (flash_vla_kernels.so) must be built separately:
    mkdir build && cd build
    cmake .. && make -j$(nproc) && make install
    cd ..

The .so is installed to flash_vla/ and is included in the package.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
