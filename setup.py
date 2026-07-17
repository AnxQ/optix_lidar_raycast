import os
import sys
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install


class CMakeExtension(Extension):
    """Extension that uses CMake to build."""
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build extension that uses CMake."""
    
    def run(self):
        """Run CMake build."""
        # Check if CMake is available
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build this package. "
                "Install with: pip install cmake"
            )
        
        # Check if CUDA is available
        try:
            subprocess.check_output(['nvcc', '--version'])
        except OSError:
            raise RuntimeError(
                "CUDA toolkit (nvcc) must be installed to build this package. "
                "Install CUDA toolkit (e.g. conda install -c nvidia cuda-nvcc)."
            )
        
        for ext in self.extensions:
            self.build_extension(ext)
    
    def build_extension(self, ext):
        """Build a single extension using CMake."""
        import pybind11

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Create build directory
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        
        # CMake configuration arguments
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPython3_EXECUTABLE={sys.executable}',
            f'-Dpybind11_DIR={pybind11.get_cmake_dir()}',
            '-DCMAKE_BUILD_TYPE=Release',
        ]
        
        # Check for OptiX SDK path
        optix_paths = [
            '/mnt/private/NVIDIA-OptiX-SDK-8.1.0-linux64-x86_64',
            os.environ.get('OptiX_INSTALL_DIR', ''),
        ]
        
        optix_path = None
        for path in optix_paths:
            if path and os.path.exists(os.path.join(path, 'include', 'optix.h')):
                optix_path = path
                break
        
        if optix_path:
            cmake_args.append(f'-DOptiX_INSTALL_DIR={optix_path}')
            print(f"Found OptiX SDK at: {optix_path}")
        else:
            raise RuntimeError(
                "OptiX SDK not found. Please set OptiX_INSTALL_DIR environment variable "
                "or install OptiX SDK to /mnt/private/NVIDIA-OptiX-SDK-8.1.0-linux64-x86_64"
            )
        
        # Build arguments
        build_args = ['--config', 'Release']
        
        # Add parallel build flag
        build_args += ['--', '-j4']
        
        # Run CMake configuration
        print("Running CMake configuration...")
        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args,
            cwd=self.build_temp
        )
        
        # Run CMake build
        print("Running CMake build...")
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args,
            cwd=self.build_temp
        )
        
        # Copy PTX file to package directory
        ptx_src = os.path.join(self.build_temp, 'optix_kernels.ptx')
        ptx_dst = os.path.join(extdir, 'optix_kernels.ptx')
        if os.path.exists(ptx_src):
            print(f"Copying PTX file: {ptx_src} -> {ptx_dst}")
            shutil.copy(ptx_src, ptx_dst)
        else:
            raise RuntimeError(f"PTX file not found at {ptx_src}")
        
        # Copy Python wrapper only if source and destination are different
        py_src = os.path.join(ext.sourcedir, 'python', 'lidar_raycast.py')
        py_dst = os.path.join(extdir, 'lidar_raycast.py')
        if os.path.exists(py_src) and os.path.abspath(py_src) != os.path.abspath(py_dst):
            print(f"Copying Python wrapper: {py_src} -> {py_dst}")
            shutil.copy(py_src, py_dst)
        
        # Create __init__.py if it doesn't exist
        init_file = os.path.join(extdir, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('from .lidar_raycast import OptiXLidarSimulator\n')
                f.write('__version__ = "0.1.0"\n')
                f.write('__all__ = ["OptiXLidarSimulator"]\n')


class CustomInstall(install):
    """Custom install command to ensure build runs first."""
    def run(self):
        self.run_command('build_ext')
        install.run(self)


# Read the README
long_description = ""
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding='utf-8')

setup(
    name='optix_lidar_raycast',
    version='0.1.0',
    author='DAPT Team',
    description='OptiX-based LiDAR raycasting simulator for batch mesh inference',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AnxQ/optix_lidar_raycast',
    
    # Package configuration
    packages=['optix_lidar_raycast'],
    package_dir={'optix_lidar_raycast': 'python'},
    ext_modules=[CMakeExtension('optix_lidar_raycast._optix_lidar_raycast')],
    
    # Build configuration
    cmdclass={
        'build_ext': CMakeBuild,
        'install': CustomInstall,
    },
    
    # Include PTX and Python files
    package_data={
        'optix_lidar_raycast': ['*.ptx', '*.py', '*.pyi', 'py.typed'],
    },
    
    zip_safe=False,
    python_requires=">=3.7",
    
    # Dependencies
    install_requires=[
        'numpy>=1.19.0',
        'torch>=1.8.0',
    ],
    
    setup_requires=[
        'cmake>=3.18',
        'pybind11>=2.6.0',
    ],

    # Non-Python build-time dependency (validated in CMakeBuild.run): nvcc
    # must be available from CUDA toolkit on PATH.
    
    # Classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
    ],
    
    keywords='lidar raycasting optix cuda simulation point-cloud',
)
