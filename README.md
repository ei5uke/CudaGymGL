# CudaGymGL
**Running thousands of training simulations in parallel, on a GPU.**

<p align="center"><img src="assets/cartpole_grid.gif" width="600" alt="Grid of 24 CartPole simulations running at once"></p>
<p align="center"><sub>Many copies of one simulation running side by side.</sub></p>

## 1. Background: how AI learns by trial and error
In **reinforcement learning (RL)**, an AI "agent" learns a task by trying actions in a simulation and getting a score back. Example: balance a pole on a moving cart (**CartPole**, shown above).

<p align="center"><img src="assets/rl_loop.svg" width="560" alt="Agent sends an action to the simulation, which sends back what happened and a score"></p>

The agent needs **millions** of these tries to learn. So the speed of the simulation matters a lot.

## 2. The problem: data keeps crossing between CPU and GPU
The AI model usually trains on the **GPU**, but popular RL simulations (like the ones in the **Gymnasium** library) run on the **CPU**. So every step, data must be copied back and forth: simulation results go to the GPU, and the model's actions come back to the CPU. Each copy is slow, and it happens millions of times.

<p align="center"><img src="assets/cpu_gpu_today.svg" width="640" alt="Today: simulation on the CPU and training on the GPU, with data copied between them every step"></p>

## 3. Our approach: put the simulation on the GPU too
<p align="center"><img src="assets/gpu_only.svg" width="640" alt="Goal: simulation and training both on the GPU, no copying between chips"></p>

- **Write the sim in CUDA:** Now both simulation and training run on the GPU.
- **Run thousands of copies in parallel:** Each GPU thread runs one copy of the simulation.
- **Retain modularity:** Wrap the code similar to a normal Gymnasium environment.
- **Visualize the environment:** An OpenGL window renders the sim in real time (like the GIF above).

Simulations: CartPole and Acrobot (a swinging two-link arm). Stretch goal: balls bouncing in a box.

## 4. Expected results
- **Speed:** steps per second as the number of copies grows from 1 to 1 million. Compare the normal CPU version against our all-GPU version. We expect the GPU to pull far ahead **ONLY** at large counts. This is because these Gymnasium toy environments are actually very simple for the CPU to produce millions of parallel copies. Herein, GPU parallelization should not truly accelerate learning much.
- **Copying cost:** measure how much time CPU-to-GPU copying takes per step, and how much we save by removing it.
- **Correctness:** GPU results match the original Gymnasium simulation step for step.
- **Demo:** a live window showing many simulations at once.