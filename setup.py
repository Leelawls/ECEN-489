from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="conv2d_cuda",
    ext_modules=[
        CUDAExtension(
            "conv2d_cuda",
            ["conv2d_cuda.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
#python setup.py install
