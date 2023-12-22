import io
import os
import re
import subprocess

from packaging.version import parse, Version
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

ROOT_DIR = os.path.dirname(__file__)

MAIN_CUDA_VERSION = "12.2"

def get_hipcc_rocm_version():
    # Run the hipcc --version command
    result = subprocess.run(['hipcc', '--version'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True)

    # Check if the command was executed successfully
    if result.returncode != 0:
        print("Error running 'hipcc --version'")
        return None

    # Extract the version using a regular expression
    match = re.search(r'HIP version: (\S+)', result.stdout)
    if match:
        # Return the version string
        return match.group(1)
    else:
        print("Could not find HIP version in the output")
        return None


def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version

def get_requirements():
    if ROCM_HOME is not None:
        req_file = 'requirements-amd.txt'
    elif CUDA_HOME is not None:
        cuda_version = get_nvcc_cuda_version(CUDA_HOME)
        if cuda_version == Version("11.8"):
            req_file = 'requirements-cu118.txt'
        else:
            req_file = 'requirements.txt'
    else:
        req_file = 'requirements-cpu.txt'
    
    with open(req_file) as f:
        requirements = f.read().splitlines()
    return requirements


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def find_version(filepath: str) -> str:
    """Extract version information from the given filepath.

    Adapted from https://github.com/ray-project/ray/blob/0b190ee1160eeca9796bc091e07eaebf4c85b511/python/setup.py
    """
    with open(filepath) as fp:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                  fp.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


def get_tabbyapi_version() -> str:
    version = find_version(get_path("tabbyapi", "__init__.py"))
    
    if ROCM_HOME is not None:
        # get the HIP version
        hipcc_version = get_hipcc_rocm_version()
        if hipcc_version is not None and hipcc_version != MAIN_CUDA_VERSION:
            rocm_version_str = hipcc_version.replace(".", "")[:3]
            version += f"+rocm{rocm_version_str}"
    elif CUDA_HOME is not None:
        cuda_version = get_nvcc_cuda_version(CUDA_HOME)
        if cuda_version is not None:
            cuda_version_str = str(cuda_version)
            # Split the version into numerical and suffix parts
            version_parts = version.split('-')
            version_num = version_parts[0]
            version_suffix = version_parts[1] if len(version_parts) > 1 else ''
            
            if cuda_version_str != MAIN_CUDA_VERSION:
                cuda_version_str = cuda_version_str.replace(".", "")[:3]
                version_num += f"+cu{cuda_version_str}"
            
            # Reassemble the version string with the suffix, if any
            version = version_num + ('-' +
                                     version_suffix if version_suffix else '')
    else:
        version += "+cpu"
        
    return version

def read_readme() -> str:
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()
    else:
        return ""

setup(
    name="tabbyapi",
    version=find_version(get_path("tabbyapi", "__init__.py")),
    description="An OAI compatible exllamav2 API that's both lightweight and fast.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="The Royal Lab",
    url="https://github.com/theroyallab/tabbyAPI",
    license='AGPL 3.0',
    packages=find_packages(exclude=["tests", "examples",
                                    "models", "loras",
                                    "templates", "Docker"]),
    install_requires=get_requirements(),
    python_requires='>=3.10, <3.12',
    entry_points={
        'console_scripts': [
            'tabbyapi=tabbyapi.main:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    # package_data={"tabbyapi": ["config.yml"]},
)