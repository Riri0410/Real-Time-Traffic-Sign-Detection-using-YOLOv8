import os
import cv2
from ultralytics import YOLO

# Ensure you have the correct path to the YOLO model file
model_path = 'yolov8m.pt'

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please check the path.")

# Load the YOLO model
model = YOLO(model_path)

# Directory containing the images
image_folder = '/Users/rishabhprasad/PycharmProjects/imageRecognition/Dataset/images'
# Directory to save the detection output
output_folder = '/Users/rishabhprasad/PycharmProjects/imageRecognition/DetectionOutput'  # Replace with your writable directory

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through each image in the directory
for image_name in os.listdir(image_folder):
    if image_name.endswith(('.png', '.jpg', '.jpeg')):
        # Load the image
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)

        # Check if the image is loaded correctly
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Run the YOLO model on the image
        results = model(image)

        # Annotate the image with bounding boxes and labels
        for result in results:
            # Iterate through each detected object
            for box in result.boxes:
                # Extract coordinates and class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{cls} {conf:.2f}"

                # Draw the bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Save the annotated image to the output folder
        output_image_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_image_path, image)

        # Optionally, you can also display the image
        # cv2.imshow('Traffic Sign Detection', image)
        # cv2.waitKey(0)

# Optionally, destroy all windows if you used cv2.imshow
# cv2.destroyAllWindows()
