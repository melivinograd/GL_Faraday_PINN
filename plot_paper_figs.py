import numpy as np
import glob
import matplotlib.pyplot as plt
import re

# Function to extract step number from filename
def extract_step_number(filename):
    match = re.search(r'step_(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return None

# Load and sort files
dt = 0.000001
gamma_0 = 0.90
L = 100
file_pattern = f'psi_{int(gamma_0*100)}_{dt}/psi_step_*.npy'
#  file_pattern = f'psi_{int(gamma_0*100)}/psi_step_*.npy'
files = sorted(glob.glob(file_pattern), key=extract_step_number)
files = files[::10]
#files = files[:100]
print(len(files))

# Load data
data = np.array([np.load(f) for f in files])

# Time step and real time calculation
timesteps = np.array([extract_step_number(f) for f in files])
real_times = timesteps * dt
print(real_times)
T = extract_step_number(files[-1])*dt

# Plot spatiotemporal plot
plt.figure(figsize=(10, 6))
plt.imshow(np.abs(data), aspect='auto', cmap='viridis', extent=[ -L/2, L/2, T, 0,])
#  plt.imshow(np.abs(data), aspect='auto', cmap='viridis')
plt.colorbar(label='Psi')
plt.xlabel('x')
plt.ylabel('Time')
#  num_ticks = 10
#  ytick_positions = np.linspace(0, len(files) - 1, num_ticks, dtype=int)
#  ytick_labels = [f'{real_times[pos]:.0f}' for pos in ytick_positions]
#  plt.yticks(ticks=ytick_positions, labels=ytick_labels)
plt.title(fr'$\gamma_0 = {gamma_0}$')
plt.show()

# Plot equispaced times
num_plots = 5
equispaced_indices = np.linspace(0, len(files) - 1, num_plots, dtype=int)
equispaced_times = real_times[equispaced_indices]

plt.figure(figsize=(15, 10))
for i, index in enumerate(equispaced_indices):
    plt.subplot(num_plots, 1, i + 1)
    plt.plot(data[index],color='grey', lw=2)
    plt.title(f'Time = {equispaced_times[i]:.0f}')
    plt.xlabel('x')
    plt.ylabel('Psi')
plt.suptitle(fr'$\gamma_0 = {gamma_0}$')
plt.tight_layout()
plt.show()#
