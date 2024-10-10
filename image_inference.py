from ultralytics import YOLO
import cv2
import supervision as sv
import matplotlib.pyplot as plt

model = YOLO("path/to/yolov5_trained_model.pt")

image = cv2.imread("path/to/img.jpg")

results = model(image)[0]

detections = sv.Detections.from_yolov8(results)

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis(False)
plt.show()

output_path = "path/to/output.jpg"
cv2.imwrite(output_path, annotated_image)
print(f"Image saved in {output_path}")
