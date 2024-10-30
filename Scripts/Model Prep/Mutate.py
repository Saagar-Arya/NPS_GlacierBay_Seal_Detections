from ultralytics import YOLO
import time

start_time = time.time()

# Load a new model from scratch
model = YOLO('yolov8s-seg.yaml')  # Adjust this to your model configuration

if __name__ == '__main__':
    # Perform hyperparameter tuning (mutate) for 30 iterations
    mutation_results = model.tune(
        data=r'C:\Users\sa553\Desktop\NPS\Crops_Recombined\combined_training_set\YOLODataset\dataset.yaml',
        epochs=30,               # Mutate for 30 epochs
        imgsz=640,
        batch=32,                # Adjusting batch size to balance memory and performance
        name="CombinedV17_Mutate",
        project=r'C:\Users\sa553\Desktop\NPS\Crops_Recombined\Models\Combined',
        workers=8,
        device='0',
        iterations=30,           # Number of generations (iterations) to run
        val=True,
    )

    # Retrieve the best model and hyperparameters after mutation
    best_model = YOLO(mutation_results['best'])  # Load the best model from mutation

    # Train the best model with optimized parameters for 100-150 epochs
    final_results = best_model.train(
        data=r'C:\Users\sa553\Desktop\NPS\Crops_Recombined\combined_training_set\YOLODataset\dataset.yaml',
        epochs=200,              
        imgsz=640,
        batch=64,                # Adjust batch size to balance memory
        name="CombinedV17_Final",
        project=r'C:\Users\sa553\Desktop\NPS\Crops_Recombined\Models\Combined',
        workers=8,
        device='0',
        cos_lr=True,             # Cosine learning rate scheduling
        val=True,
        plots=True,              # Enable plots during final training
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Time script took to run = {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds.")
