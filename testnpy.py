import numpy as np
import matplotlib.pyplot as plt

actions = np.load("output/actions.npy")   # (20, 7)
labels = ["delta_x", "delta_y", "delta_z", "delta_roll", "delta_pitch", "delta_yaw", "gripper"]

plt.figure(figsize=(12, 4))
for i, label in enumerate(labels):
    plt.plot(actions[:, i], label=label)
plt.axhline(1, color="k", linestyle="--", alpha=0.3)
plt.axhline(-1, color="k", linestyle="--", alpha=0.3)
plt.legend(loc="upper right")
plt.xlabel("step")
plt.ylabel("action value")
plt.title("SmolVLA action trajectory (random expert)")
plt.tight_layout()
plt.savefig("output/action_plot.png")