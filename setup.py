import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)


if __name__ == '__main__':

    setup(
        name='quantization',
        version="0.0.0",
        description='',
        install_requires=[
            'numpy',
            'torch>=1.1',
            'spconv',
            'numba',
            'easydict',
            'pyyaml'
        ],
        author='QuanVu',
        packages=find_packages(exclude=['tools', 'data', 'output']),
        cmdclass={'build_ext': BuildExtension},
        ext_modules=[
            make_cuda_ext(
                name='iou3d_nms_cuda',
                module='core.ops.iou3d_nms',
                sources=[
                    'src/iou3d_cpu.cpp',
                    'src/iou3d_nms_api.cpp',
                    'src/iou3d_nms.cpp',
                    'src/iou3d_nms_kernel.cu',
                ]
            ),

            CUDAExtension(
                name="core.ops.nms.nms",
                sources=[
                    "core/ops/nms/nms.cc",
                    "core/ops/nms/nms_kernel.cu",
                ],
                extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
            ),
            make_cuda_ext(
                name='pointnet2_stack_cuda',
                module='core.ops.pointnet2.pointnet2_stack',
                sources=[
                    'src/pointnet2_api.cpp',
                    'src/ball_query.cpp',
                    'src/ball_query_gpu.cu',
                    'src/group_points.cpp',
                    'src/group_points_gpu.cu',
                    'src/sampling.cpp',
                    'src/sampling_gpu.cu', 
                    'src/interpolate.cpp', 
                    'src/interpolate_gpu.cu',
                    'src/voxel_query.cpp', 
                    'src/voxel_query_gpu.cu',
                    'src/vector_pool.cpp',
                    'src/vector_pool_gpu.cu'
                ],
            ),
            make_cuda_ext(
                name='pointnet2_batch_cuda',
                module='core.ops.pointnet2.pointnet2_batch',
                sources=[
                    'src/pointnet2_api.cpp',
                    'src/ball_query.cpp',
                    'src/ball_query_gpu.cu',
                    'src/group_points.cpp',
                    'src/group_points_gpu.cu',
                    'src/interpolate.cpp',
                    'src/interpolate_gpu.cu',
                    'src/sampling.cpp',
                    'src/sampling_gpu.cu',

                ],
            ),

            make_cuda_ext(

                name='roiaware_pool3d_cuda',
                module='core.ops.roiaware_pool3d',
                sources=[
                    'src/roiaware_pool3d.cpp',
                    'src/roiaware_pool3d_kernel.cu',
                ]
            ),
        ],
    )
