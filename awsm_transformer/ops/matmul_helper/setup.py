import os
import subprocess
import sys
import shutil
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        # 1. Determine build directory
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        # 2. CMake Arguments
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DCMAKE_BUILD_TYPE=Debug',
        ]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # 3. Run CMake
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'], cwd=self.build_temp)

        # 4. ROBUST COPY: Copy generated library to source tree for editable installs
        lib_name = "libmatmul_dispatcher"
        built_file = None
        
        # Look for the built file in the build_temp directory or where CMake put it
        # Because CMakeLists.txt might force output to source, we check both locations
        possible_locs = [extdir, os.path.join(os.path.dirname(os.path.abspath(__file__)), "matmul_dispatcher")]
        
        for loc in possible_locs:
            if not os.path.exists(loc): continue
            for f in os.listdir(loc):
                if lib_name in f and (f.endswith('.so') or f.endswith('.dll') or f.endswith('.dylib')):
                    built_file = os.path.join(loc, f)
                    break
            if built_file: break

        if built_file:
            dest_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matmul_dispatcher")
            dest_path = os.path.join(dest_dir, os.path.basename(built_file))
            
            # --- FIX: Only copy if source and destination are different ---
            try:
                if os.path.abspath(built_file) != os.path.abspath(dest_path):
                    print(f"Copying {built_file} -> {dest_dir}")
                    shutil.copy(built_file, dest_dir)
                else:
                    print(f"Library already in correct location: {built_file}")
            except shutil.SameFileError:
                pass # Explicitly ignore if they are the same file

setup(
    name='matmul_dispatcher',
    version='0.0.1',
    packages=find_packages(),
    package_data={'matmul_dispatcher': ['*.so', '*.dll', '*.dylib']}, 
    ext_modules=[CMakeExtension('matmul_dispatcher.libmatmul_dispatcher')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    install_requires=['torch'],
)
