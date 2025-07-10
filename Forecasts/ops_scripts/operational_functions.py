import os

def load_strings_from_file_list(file_path):
    with open(file_path, 'r') as f:
        return [str(line.strip()) for line in f]

# Load the checkpoint (starting point)
def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return str(f.read().strip())  # Return the last saved index
    return 0  # Start from the beginning if no checkpoint exists

# Save the current progress to a checkpoint file
def save_checkpoint(index, checkpoint_file):
    with open(checkpoint_file, 'w') as f:
        f.write(str(index))