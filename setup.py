from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="conv2d_cuda",
    ext_modules=[
        CUDAExtension(
            "conv2d_cuda",
            ["conv2d_cuda.cu"],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": [
                    "-O2",
                    "--gpu-architecture=compute_86",  # Use compute_75 for CUDA 10.2
                    "--gpu-code=sm_86"
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
