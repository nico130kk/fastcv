from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fastcv",
    ext_modules=[
        CUDAExtension(
            name="fastcv",
            sources=[
                "kernels/grayscale.cu",
                "kernels/box_blur.cu",
                "kernels/sobel.cu",
                "kernels/dilation.cu",
                "kernels/erosion.cu",
                "kernels/pyrDown.cu",
                "kernels/module.cpp",
            ],
            extra_compile_args={"cxx": ["-O2"],
            'nvcc': ['-O2', '-allow-unsupported-compiler', '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH']
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
