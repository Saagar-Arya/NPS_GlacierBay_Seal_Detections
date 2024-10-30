from ultralytics import YOLO
import yaml
import time

start_time = time.time()

# Load a new model with the desired configuration
model = YOLO('yolov8s-seg.yaml')  # Adjust this to your model configuration file path if necessary

# Path to the best hyperparameters file
best_hyperparams_path = r'C:\Users\sa553\Desktop\NPS\Crops_Recombined\Models\Combined\tune\best_hyperparameters.yaml'

# Load the best hyperparameters
with open(best_hyperparams_path, 'r') as file:
    best_hyperparameters = yaml.safe_load(file)
if __name__ == '__main__':

    # Train the model with the best hyperparameters
    final_results = model.train(
        data=r'C:\Users\sa553\Desktop\NPS\Crops_Recombined\inverse_training_set\YOLODataset\dataset.yaml',
        epochs=200,               # Adjust epochs as needed
        imgsz=640,
        batch=64,                 # Adjust batch size to balance memory
        name="InverseV17_Final",
        project=r'C:\Users\sa553\Desktop\NPS\Crops_Recombined\Models\Inverse',
        workers=8,
        device='0',
        cos_lr=True,              # Use cosine learning rate scheduling
        val=True,
        plots=True,               # Enable plots during final training
        **best_hyperparameters    # Unpack and apply hyperparameters
    )

# Calculate and print elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Time script took to run = {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds.")
