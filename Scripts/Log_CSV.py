import csv
import os

def log_to_csv(log_filename):
    # Define the CSV file path by joining the directory and the desired file name
    csv_directory = os.path.dirname(log_filename)
    name = os.path.basename(log_filename)
    name = os.path.splitext(name)[0] +".csv"
    csv_filename = os.path.join(csv_directory, name)
    
    with open(log_filename, 'r') as log_file, open(csv_filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write CSV headers
        csv_writer.writerow(["Image Filename", "Number of Seals", "Bounding Boxes"])

        # Skip the first line of the log file
        next(log_file)

        for line in log_file:
            line = line.strip()
            if not line:
                continue

            # Split the line by commas to separate image name and bounding boxes
            parts = line.split(',', 1)
            image_name = parts[0].strip()
            if len(parts) > 1:
                # Bounding boxes present
                bounding_boxes_str = parts[1].strip()
                bounding_boxes = bounding_boxes_str.split(' ')
                # Group every 4 values (representing a bounding box) together into a single string
                grouped_bounding_boxes = [' '.join(bounding_boxes[i:i+4]) for i in range(0, len(bounding_boxes), 4)]
                num_seals = len(grouped_bounding_boxes)
            else:
                # No bounding boxes, hence no seals
                grouped_bounding_boxes = []
                num_seals = 0

            # Prepare row data for CSV
            row = [image_name, num_seals] + grouped_bounding_boxes
            csv_writer.writerow(row)

    print(f"CSV file created successfully at: {csv_filename}")

if __name__ == "__main__":
    log_file_path = r''
    log_to_csv(log_file_path)