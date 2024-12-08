from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np

# Paths to TensorBoard log files and corresponding experiment names
log_paths = [
    "./runs/experiment_20241207_131809/events.out.tfevents.1733606289.AMDITX.40297.0",
    "./runs/experiment_20241207_131822/events.out.tfevents.1733606302.AMDITX.40701.0",
    "./runs/experiment_20241207_153636/events.out.tfevents.1733614596.AMDITX.121025.0",
    "./runs/experiment_20241207_153655/events.out.tfevents.1733614615.AMDITX.121398.0"
]

# Corresponding experiment names for the legend
experiment_names = ["tau=0.001", "tau=0.01", "tau=0.025", "tau=0.05"]

# Smoothing function: Moving Average
def moving_average(values, window_size=10):
    """Apply a simple moving average to smooth the data."""
    if len(values) < window_size:
        return values  # Return as-is if not enough points
    return np.convolve(values, np.ones(window_size) / window_size, mode='valid')

# Dictionary to store data from logs
data = {}

# Extract scalar data for Loss from logs
for path in log_paths:
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()  # Load data from the log
    scalars = ea.Scalars("Loss")  # Use the correct tag name
    steps = [scalar.step for scalar in scalars]
    values = [scalar.value for scalar in scalars]
    data[path] = (steps, values)

# Plot the data
plt.figure(figsize=(10, 6))
for path, (steps, values), name in zip(log_paths, data.values(), experiment_names):
    # Normalize x-axis to millions of steps
    steps_in_millions = [step / 1_000_000 for step in steps]
    # Apply smoothing to y-axis values
    smoothed_values = moving_average(values)
    smoothed_steps = steps_in_millions[:len(smoothed_values)]  # Adjust steps to match smoothed values
    # Cap y-axis values to 5
    capped_values = [min(value, 5) for value in smoothed_values]
    plt.plot(smoothed_steps, capped_values, label=name)  # Use experiment name for label

# Formatting for IEEE paper
plt.xlabel("Steps (in millions)", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Loss Convergence for Values of Tau", fontsize=14)
plt.legend(loc="upper right", fontsize=10)
plt.ylim(0, 5)  # Set y-axis range to [0, 5]
plt.grid(True)
plt.tight_layout()

# Save the plot as a high-resolution image for IEEE paper
plt.savefig("loss_convergence_smoothed.png", dpi=300)
plt.show()
