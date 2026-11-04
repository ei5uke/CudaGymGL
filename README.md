# CudaGymGL
GPU-Batched Physics Environments for RL, with Real-Time OpenGL Rendering

CSCI596 final project (solo). Simple Gymnasium-style physics environments written as CUDA C kernels, where one GPU thread simulates one environment copy, so thousands to millions of copies step in parallel. A thin Python wrapper exposes them as a `gymnasium.vector.VectorEnv`, and an OpenGL viewer animates many copies at once.

## Plan

**Environments**
- CartPole (same equations and constants as Gymnasium's `CartPole-v1`)
- Acrobot (two-link pendulum, RK4 integrator)
- Balls-in-box (stretch goal): 2D elastic collisions, MD-like pair loops

**Components**
1. **CUDA kernels** (`.cu`): one `__global__` step kernel per environment, physics constants in `__constant__` memory, `cudaMalloc`/`cudaMemcpy` for state, in the style of the class assignments (`pdf.cu`, `hypi_setdevice.cu`).
2. **Serial C baseline**: the same step functions on the CPU, for speedup comparisons.
3. **Gymnasium wrapper**: kernels built into a shared library (`nvcc -shared`), loaded with `ctypes`, and wrapped as a `gymnasium.vector.VectorEnv`.
4. **OpenGL viewer**: C + GLUT, modeled on `atomv.c`, drawing a grid of environment copies in real time.
5. **Benchmarks**: Slurm scripts for USC CARC.

**Evaluation**
- Steps/sec vs number of environments N (1 to 10^6): serial C, CUDA kernel only, CUDA through the Python wrapper, and Gymnasium's CPU vector env.
- Cost of host-device transfer per step.
- Correctness: CUDA trajectories match Gymnasium for a fixed action sequence; balls-in-box conserves kinetic energy.
- Optional: multi-GPU scaling with MPI + CUDA.

## Status
Planning. Code to come.
