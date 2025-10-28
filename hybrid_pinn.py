import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Beam properties
L = 1.0  # Length of the beam
E = 1.0  # Young's modulus
I = 1.0  # Moment of inertia
rho = 1.0  # Density
A = 1.0  # Cross-sectional area
EI = E * I
m = rho * A

# Ramp force f(t) = q*t
q = 2.0

# Time domain
t_start = 0.0
t_end = 4.0
dt = 1.0  # Time step size
time_steps = np.arange(t_start, t_end + dt, dt)

# Newmark method parameters
gamma = 0.5
beta = 0.25

# PINN training parameters
# NOTE: epochs_per_step is set to a low value to allow the script to run quickly
# in a constrained environment. For accurate results, increase this to 2000 or more.
epochs_per_step = 100
learning_rate = 1e-3
num_collocation_points = 500
num_boundary_points = 100


# Define the Physics-Informed Neural Network for the spatial domain
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 100),  # Input is only x
            nn.SiLU(),
            nn.Linear(100, 100),
            nn.SiLU(),
            nn.Linear(100, 100),
            nn.SiLU(),
            nn.Linear(100, 100),
            nn.SiLU(),
            nn.Linear(100, 100),
            nn.SiLU(),
            nn.Linear(100, 100),
            nn.SiLU(),
            nn.Linear(100, 100),
            nn.SiLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        return self.net(x)

# Initialize displacement, velocity, and acceleration
# Using a grid of points for tracking the solution over time
x_eval = torch.linspace(0, L, num_collocation_points, requires_grad=True).view(-1, 1)
u_prev = torch.zeros_like(x_eval)
v_prev = torch.zeros_like(x_eval)
a_prev = torch.zeros_like(x_eval)

tip_displacements = []

# Main time-stepping loop
for i, t in enumerate(time_steps[1:], 1):  # Start from the first time step
    print(f"Solving for time step {i}/{len(time_steps)-1}, t = {t:.4f}")

    # Instantiate a new PINN for the current time step
    pinn = PINN()
    optimizer = torch.optim.Adam(pinn.parameters(), lr=learning_rate)

    # Calculate the effective stiffness matrix component from Newmark's method
    a0 = 1 / (beta * dt**2)
    a1 = 1 / (beta * dt)
    a2 = 1 / (2 * beta) - 1

    # Effective force includes terms from previous step's motion
    effective_force_terms = m * (a0 * u_prev + a1 * v_prev + a2 * a_prev)

    # Current time's ramp force
    current_force = q * t

    # Inner loop for training the PINN at the current time step
    for epoch in range(epochs_per_step):
        optimizer.zero_grad()

        # Generate collocation points for this training epoch
        x_physics = torch.rand(num_collocation_points, 1, requires_grad=True) * L
        u = pinn(x_physics)

        # Compute spatial derivatives
        u_x = torch.autograd.grad(u, x_physics, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_physics, torch.ones_like(u_x), create_graph=True)[0]
        u_xxx = torch.autograd.grad(u_xx, x_physics, torch.ones_like(u_xx), create_graph=True)[0]
        u_xxxx = torch.autograd.grad(u_xxx, x_physics, torch.ones_like(u_xxx), create_graph=True)[0]

        # Newmark's equation for acceleration at the current step: a_n+1 = a0 * (u_n+1 - u_n) - a1 * v_n - a2 * a_n
        # We need to evaluate the previous state at the same collocation points
        u_prev_interp = torch.tensor(np.interp(x_physics.detach().numpy().flatten(), x_eval.detach().numpy().flatten(), u_prev.detach().numpy().flatten()), dtype=torch.float32).view(-1, 1)
        v_prev_interp = torch.tensor(np.interp(x_physics.detach().numpy().flatten(), x_eval.detach().numpy().flatten(), v_prev.detach().numpy().flatten()), dtype=torch.float32).view(-1, 1)
        a_prev_interp = torch.tensor(np.interp(x_physics.detach().numpy().flatten(), x_eval.detach().numpy().flatten(), a_prev.detach().numpy().flatten()), dtype=torch.float32).view(-1, 1)

        # PDE loss based on the rearranged governing equation
        # EI * u_xxxx + m * a_n+1 = f(t_n+1)
        # EI * u_xxxx + m * (a0 * (u - u_prev) - a1*v_prev - a2*a_prev) = f(t)
        pde_residual = EI * u_xxxx + m * (a0 * (u - u_prev_interp) - a1 * v_prev_interp - a2 * a_prev_interp) - current_force
        loss_pde = torch.mean(pde_residual**2)

        # Boundary conditions (cantilever beam)
        # At x = 0 (fixed end): u = 0, u_x = 0
        x_bc_fixed = torch.zeros(num_boundary_points, 1, requires_grad=True)
        u_fixed = pinn(x_bc_fixed)
        u_fixed_x = torch.autograd.grad(pinn(x_bc_fixed), x_bc_fixed, torch.ones_like(u_fixed), create_graph=True)[0]
        loss_bc_fixed = torch.mean(u_fixed**2) + torch.mean(u_fixed_x**2)

        # At x = L (free end): u_xx = 0, u_xxx = 0
        x_bc_free = torch.ones(num_boundary_points, 1, requires_grad=True) * L
        u_free = pinn(x_bc_free)
        u_free_x = torch.autograd.grad(u_free, x_bc_free, torch.ones_like(u_free), create_graph=True, retain_graph=True)[0]
        u_free_xx = torch.autograd.grad(u_free_x, x_bc_free, torch.ones_like(u_free_x), create_graph=True)[0]
        u_free_xxx = torch.autograd.grad(u_free_xx, x_bc_free, torch.ones_like(u_free_xx), create_graph=True)[0]
        loss_bc_free = torch.mean(u_free_xx**2) + torch.mean(u_free_xxx**2)

        # Total loss
        total_loss = loss_pde + loss_bc_fixed + loss_bc_free

        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'  Epoch [{epoch+1}/{epochs_per_step}], Loss: {total_loss.item():.6f}')

    # After training, get the displacement at the current time step
    u_current = pinn(x_eval).detach()

    # Update acceleration and velocity using Newmark's formulas
    a_current = a0 * (u_current - u_prev) - a1 * v_prev - a2 * a_prev
    v_current = v_prev + (1 - gamma) * dt * a_prev + gamma * dt * a_current

    # Store the tip displacement
    tip_displacement = pinn(torch.tensor([[L]], dtype=torch.float32)).detach().item()
    tip_displacements.append(tip_displacement)

    # Update the state for the next time step
    u_prev, v_prev, a_prev = u_current, v_current, a_current

# Plotting the tip displacement over time
plt.figure(figsize=(10, 6))
plt.plot(time_steps[1:], tip_displacements, label='PINN Tip Displacement')
plt.title('Tip Displacement of Cantilever Beam Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Displacement')
plt.grid(True)
plt.legend()
plt.savefig('tip_displacement.png')
plt.show()
