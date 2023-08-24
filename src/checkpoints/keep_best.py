import os

def get_validation_loss(checkpoint_folder):
    # Extract validation loss from the checkpoint folder name
    return float(checkpoint_folder.split("_")[-1])

def keep_best_checkpoint(checkpoint_folder):
    # Get a list of all checkpoint folders
    checkpoint_folders = [folder for folder in os.listdir(checkpoint_folder) if os.path.isdir(os.path.join(checkpoint_folder, folder))]
    
    if not checkpoint_folders:
        print("No checkpoint folders found.")
        return
    
    # Find the checkpoint with the lowest validation loss
    best_checkpoint = min(checkpoint_folders, key=get_validation_loss)
    
    # Keep the best checkpoint and delete the others
    for folder in checkpoint_folders:
        if folder == best_checkpoint:
            print(f"Keeping checkpoint: {folder}")
        else:
            folder_path = os.path.join(checkpoint_folder, folder)
            print(f"Deleting checkpoint: {folder}")
            # Delete the folder and its contents
            os.system(f"rm -rf {folder_path}")

if __name__ == "__main__":
    checkpoint_folder = os.path.dirname(os.path.abspath(__file__))
    keep_best_checkpoint(checkpoint_folder)
