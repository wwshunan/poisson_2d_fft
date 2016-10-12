import numpy as np
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from scipy.stats import multivariate_normal
import skcuda.fft as cu_fft


mod = SourceModule("""
    #include "stdio.h"
    const double PI = 3.141592653589793238463;
    __global__ void rearrange(double *qGrid, double *dest, int row_len)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col == 0){
            dest[row * row_len + col] = 0;
            }
        else {
            dest[row * row_len + col] = sin(col * PI / row_len) * (qGrid[row * row_len + col] + qGrid[row * row_len + row_len - col]) + 0.5 * (qGrid[row * row_len + col] - qGrid[row * row_len + row_len - col]);
            }
    }
    """)
xGridBase = 4
yGridBase = 4
x, y = np.mgrid[-1.0:1.0:xGridBase * 1j, -1.0:1.0:yGridBase * 1j]
xy = np.column_stack([x.flat, y.flat])

mu = np.array([0.0, 0.0])

sigma = np.array([.25, .25])
covariance = np.diag(sigma**2)

z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
print z
z_gpu = gpuarray.to_gpu(z)
dest = np.zeros_like(z)
#dest = gpuarray.zeros_like(z_gpu)
rearrange = mod.get_function("rearrange")
rearrange(z_gpu, cuda.Out(dest), np.int32(yGridBase), block=(xGridBase, yGridBase, 1))
z_col_forward = gpuarray.empty((xGridBase, yGridBase//2+1), np.complex128)
print dest

dest_gpu = gpuarray.to_gpu(dest)

#z_col = np.empty(yGridBase, dtype=np.float32)
#z_col_gpu = gpuarray.empty(yGridBase, np.float32)
#z_col_gpu_dest = gpuarray.empty(yGridBase//2+1, np.complex64)
plan_forward = cu_fft.Plan((xGridBase, yGridBase), np.float64, np.complex128)
cu_fft.fft(dest_gpu, z_col_forward, plan_forward)

print z_col_forward.get()

dest_temp = dest.reshape(8, 2)
dest_c  = dest_temp[..., 0] + 1j * dest_temp[..., 1]
dest_r = np.fft.fft2(dest_c.reshape(4, 2))
print dest_r

#z = np.zeros((32, 32), dtype=np.float32)
#cuda.memcpy_dtoh(z, z_gpu)


'''
for i in range(xGridBase):
    for i in range(1, yGridBase):
        z_col[i] = np.sin(i*np.pi/N) * (x[i] + x[yGridBase-i]) + 0.5 * (x[i] - x[yGridBase-i])
    z_col_gpu = gpuarray.to_gpu(z_col)
    cu_fft.fft(z_col_gpu, z_col_gpu_dest, plan_forward)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()

N = 16
x = np.random.randint(2, size=N-1)


y_gpu = gpuarray.to_gpu(y)
yf_gpu = gpuarray.empty(N//2+1, np.complex64)
plan_forward = cu_fft.Plan(N, np.float32, np.complex64)
cu_fft.fft(y_gpu, yf_gpu, plan_forward)
print yf_gpu
print dst(x, 1)
'''
