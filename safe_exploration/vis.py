import matplotlib.pyplot as plt
import numpy as np

# Load replay buffer
data = np.load("data/replay_buffer.npz")
actions = np.array(data["actions"])  # shape: (N, 2)
observations = np.array(data["observations"])  # shape: (N, 4)

# ---- Plot actions ----
plt.figure(figsize=(5,5))
plt.scatter(actions[:,0], actions[:,1], alpha=0.6)
plt.xlabel("Action dimension 0")
plt.ylabel("Action dimension 1")
plt.title("Scatter of Actions")
plt.xlim(-0.3, 0.3)
plt.ylim(-0.3, 0.3)
plt.grid(True)
plt.show()

# ---- Plot lidar distances over time ----
plt.figure(figsize=(10,5))
for i in range(observations.shape[1]):  # loop over the 4 lidar readings
    plt.plot(observations[:,i], label=f"Lidar {i}")
plt.xlabel("Step")
plt.ylabel("Minimum distance")
plt.title("Minimum Lidar Distances Over Steps")
plt.legend()
plt.grid(True)
plt.show()

# ---- Optional: histogram of lidar distances ----
plt.figure(figsize=(10,5))
for i in range(observations.shape[1]):
    plt.hist(observations[:,i], bins=30, alpha=0.5, label=f"Lidar {i}")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.title("Distribution of Minimum Lidar Distances")
plt.legend()
plt.show()