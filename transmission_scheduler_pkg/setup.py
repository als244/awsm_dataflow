from setuptools import setup, Extension, find_packages
import sys

# Define compiler flags
extra_compile_args = []
if sys.platform != 'win32':
    # Linux/Mac optimization flags
    extra_compile_args = ['-O3', '-fPIC']
else:
    # Windows flags (optional, if you ever run there)
    extra_compile_args = ['/O2']

# Define the C Extension
# We name it '_capi' so it sits inside the package as a hidden module
scheduler_module = Extension(
    name="transmission_scheduler._capi",
    sources=["transmission_scheduler/transmission_scheduler.c"],
    extra_compile_args=extra_compile_args,
    # We do NOT include -mavx2 here because the C code handles 
    # AVX dispatch internally via attributes. This keeps it portable!
)

setup(
    name="transmission_scheduler",
    version="0.0.1",
    description="High-performance transmission scheduler (AVX2/Scalar)",
    packages=find_packages(),
    ext_modules=[scheduler_module],
    install_requires=["numpy"],
    zip_safe=False,
)