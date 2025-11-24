import matplotlib.pyplot as plt
import numpy as np

# Load replay buffer
root_path = "data/safety_layer"
data = np.load(f"{root_path}/replay_buffer.npz")
actions = np.array(data["actions"])  # shape: (N, 2)
observations = np.array(data["observations"])  # shape: (N, Num lidars)
c = np.array(data["c"])
c_next = np.array(data["c_next"])
agent_position = np.array(data["agent_position"])

def plot_actions(actions: np.ndarray, filename:str):
    # Actions is (Samples, 2)
    plt.figure(figsize=(5,5))
    plt.scatter(actions[:,0], actions[:,1], alpha=0.6, s=0.5)
    plt.xlabel("Action dimension 0")
    plt.ylabel("Action dimension 1")
    plt.title("Scatter of Actions")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

def plot_action_histogram(actions: np.ndarray, filename: str):
    # actions: shape (Samples, 2)
    plt.figure(figsize=(6,5))
    plt.hist2d(actions[:,0], actions[:,1], bins=50, cmap='viridis')
    plt.colorbar(label='Frequency')
    plt.xlabel("Action dimension 0")
    plt.ylabel("Action dimension 1")
    plt.title("2D Histogram of Actions")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

def plot_observations(observations: np.ndarray, filename:str):
    plt.figure(figsize=(10,5))
    plt.scatter(np.arange(0, observations.shape[0],1), observations[:,0], label=f"Lidar {0}")
    #for i in range(observations.shape[1]): 
    #    plt.plot(observations[:,i], label=f"Lidar {i}")
    plt.xlabel("Step")
    plt.ylabel("Minimum distance")
    plt.title("Minimum Lidar Distances Over Steps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)  

def plot_observation_histogram(observations: np.ndarray, filename: str):
    plt.figure(figsize=(6,5))
    plt.hist(observations[:,0], bins=50, color='steelblue', alpha=0.8)
    plt.xlabel("Minimum Lidar Distance")
    plt.ylabel("Frequency")
    plt.title("Histogram of Minimum Lidar Distances")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

def plot_observation_histogram_grid(observations: np.ndarray, filename: str):
    num_features = observations.shape[1]
    
    # Dynamically determine grid size (square-ish layout)
    cols = int(np.ceil(np.sqrt(num_features)))
    rows = int(np.ceil(num_features / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = np.array(axes).reshape(-1)  # Flatten in case it's 2D

    for i in range(num_features):
        ax = axes[i]
        ax.hist(observations[:, i], bins=50, color='steelblue', alpha=0.8)
        ax.set_title(f"Lidar {i}")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Frequency")
        ax.grid(True)

    # Hide any unused subplots if num_features < rows*cols
    for j in range(num_features, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Histograms of Lidar Observations", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(filename, dpi=300)
    plt.show()

# def plot_observation_histogram_grid(observations: np.ndarray, filename: str):
#     # Dynamically determine grid size (square-ish layout)

#     fig, ax = plt.subplots(figsize=(8, 7))

#     ax.hist(observations, bins=5, color='steelblue', alpha=0.8)
#     print(observations)
#     print(np.min(observations), np.max(observations))
#     ax.set_title(f"Lidar 0")
#     ax.set_xlabel("Distance")
#     ax.set_ylabel("Frequency")
#     ax.grid(True)

#     fig.suptitle("Histograms of Lidar Observations", fontsize=14)
#     plt.tight_layout(rect=[0, 0, 1, 0.97])
#     plt.savefig(filename, dpi=300)
#     plt.show()

def plot_c(c: np.ndarray, filename:str):
    plt.figure(figsize=(10,5))
    plt.scatter(np.arange(0, c.shape[0], 1), c[:,0], label=f"Lidar {0}")
    plt.xlabel("Step")
    plt.ylabel("Constraint Values")
    plt.title("Constraints")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)  

def plot_c_histogram(c: np.ndarray, filename: str):
    plt.figure(figsize=(6,5))
    plt.hist(c[:,0], bins=50, color='darkorange', alpha=0.8)
    plt.xlabel("Constraint Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Constraint Values")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

def plot_agent_position(agent_position: np.ndarray, filename:str):
    # Actions is (Samples, 2)
    plt.figure(figsize=(5,5))
    plt.scatter(agent_position[:,0], agent_position[:,1], alpha=0.6, s=0.1)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter of Agent Positions")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

def plot_agent_position_histogram(agent_position: np.ndarray, filename: str):
    # actions: shape (Samples, 2)
    plt.figure(figsize=(6,5))
    plt.hist2d(agent_position[:,0], agent_position[:,1], bins=50, cmap='viridis')
    plt.colorbar(label='Frequency')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D Histogram of Agent Positions")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

filename=f"{root_path}/actions_plot.jpg"
plot_actions(actions, filename)

filename=f"{root_path}/actions_plot_historgram.jpg"
plot_action_histogram(actions, filename)

filename=f"{root_path}/observations_plot.jpg"
plot_observation_histogram_grid(observations, filename)

filename=f"{root_path}/c.jpg"
print(f"Max value of c: {np.max(c)}")
plot_c_histogram(c, filename)

filename=f"{root_path}/c_next.jpg"
print(f"Max value of c_next: {np.max(c_next)}")
plot_c_histogram(c_next, filename)

filename=f"{root_path}/agent_position.jpg"
plot_agent_position(agent_position, filename)
filename=f"{root_path}/agent_position_histogram.jpg"
plot_agent_position_histogram(agent_position, filename)
