# **Glacier Bay Seal Ice Preference Analysis**

## **Overview**

This project provides a workflow for processing drone imagery from Glacier Bay National Park to analyze seal ice preference. The workflow includes scripts that detect seal locations, draw bounding boxes (optional), and analyze ice formations in relation to those locations. The final output reports both seal locations and the ice characteristics, including ice area of seals located on ice.

## **Workflow:**

- **Seal Detection**: A script processes the images to infer seal locations using the combined results of three YOLOV8 models.

- **Optional .TXT Log file to .CSV**: A script to convert the log.txt file into a more user friendly format.

- **Optional Bounding Box Visualization**: Another script can be run to draw bounding boxes around detected seals.

- **Ice Formation Analysis**: A final script identifies ice formations, compares them with seal locations, and returns ice characteristics (e.g., area of ice if a seal is on it).


## **Directory Structure**

- **CropsAll Folder**: Contains image crops and corresponding JSON files used for model training. Crops_Source contains 3 unique sets of crops. Crops_Filtered contains the combination of the 3 unique set of crops. 

Crops were generated from larger images containing seals, and the seals were traced using LabelMe and a single class "seal"

- **Models Folder**: Contains three models whose outputs are combined to produce the final seal detection result:

  1. A combined model trained on both inverse and color crops.
  2. A color model trained only on color crops.
  3. An inverse model trained only on inverse crops.
  4. These models are merged to increase confidence in seal detection during inference.

- **Sample_Images Folder**: Includes sample images of seals for testing and demonstration purposes.

- **Scripts Folder**: Contains scripts for detailed processing, including crop image augmentation for model creation and smaller step scripts for ice thresholding and tracing.


## **Usage Instructions**

Running the main script Full_System_Run.py with the sample images will:
1. **Generate a log file** of seal locations.
2. **Overlay bounding boxes** on the original images (Optional).
3. **Profile ice formations** and compare them with seal locations to report ice characteristics and area, if applicable.


## **Future Work**

1. Add padding to images to ensure seals caught between crops are not double counted
2. Adjust training data. Test by excluding Spider Reef Data and retraining all models
3. Adjust Thresholding on Ice to better include ice, exclude sun reflections, etc...
4. Consider experimenting with HSV Filtering to improve model detections by focusing on specific colors and shapes
https://stackoverflow.com/questions/57469394/opencv-choosing-hsv-thresholds-for-color-filtering
