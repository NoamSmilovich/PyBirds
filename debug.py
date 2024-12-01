import os
import re
import matplotlib.pyplot as plt

# Define directories
logs_dir = "logs/"
models_dir = "models/"
output_dir = "plots/"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Regular expressions to extract job ID and match patterns
log_pattern = re.compile(r"log-(\d+)\.out")
epoch_pattern = re.compile(r"INFO:root:Epoch \[(\d+)/\d+\] Complete")
step_loss_pattern = re.compile(
    r"INFO:root:Step \[\d+\], Training Loss: ([\d.]+), Validation Loss: ([\d.]+), Accuracy: ([\d.]+)%"
)

# Loop through log files in the logs directory
for log_file in os.listdir(logs_dir):
    log_match = log_pattern.match(log_file)
    if not log_match:
        continue

    # Extract the job ID
    job_id = log_match.group(1)
    model_folder = f"classifier_jobid={job_id}"

    # Check if the corresponding model folder exists
    if not os.path.isdir(os.path.join(models_dir, model_folder)):
        print(f"Skipping {log_file}, no corresponding model folder found.")
        continue

    # Initialize data storage
    epochs = []
    training_losses = []
    validation_losses = []
    accuracies = []

    # Parse the log file
    log_file_path = os.path.join(logs_dir, log_file)
    current_epoch = None
    with open(log_file_path, "r") as file:
        for line in file:
            # Check for epoch completion
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))

            # Check for training/validation loss and accuracy
            step_match = step_loss_pattern.search(line)
            if step_match and current_epoch is not None:
                training_loss = float(step_match.group(1))
                validation_loss = float(step_match.group(2))
                accuracy = float(step_match.group(3))
                epochs.append(current_epoch)
                training_losses.append(training_loss)
                validation_losses.append(validation_loss)
                accuracies.append(accuracy)

    # If no data was collected, skip this log file
    if not epochs:
        print(f"No relevant data found in {log_file}, skipping.")
        continue

    # Plot the results
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot training and validation losses on the left y-axis
    ax1.plot(epochs, training_losses, label="Training Loss", color='tab:blue')
    ax1.plot(epochs, validation_losses, label="Validation Loss", color='tab:orange')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='tab:blue')
    ax1.set_ylim(0, 3)  # Set the Loss y-axis range from 0 to 2
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a second y-axis to plot accuracy
    ax2 = ax1.twinx()
    ax2.plot(epochs, accuracies, label="Accuracy", color='tab:green')
    ax2.set_ylabel("Accuracy", color='tab:green')
    ax2.set_ylim(50, 100)
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # Add titles and legends
    ax1.set_title(f"Training Loss, Validation Loss, and Accuracy for Job ID {job_id}")
    ax1.legend(loc='upper left')

    # Show the plot
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, f"loss_accuracy_plot_{job_id}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Generated plot for Job ID {job_id} and saved to {plot_path}.")