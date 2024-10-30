import yolov5
import cv2
import supervision as sv
import torch

model = yolov5.load('best2.pt')

video_path = "Simpang_Gondo.mp4"
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = "output_gondo.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def custom_label(label: str, confidence: float | None) -> str:
    return f"{label} {confidence:.2f}" if confidence is not None else label

# Initialize annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

frame_count = 0
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
        results = model(frame)

    detections = sv.Detections.from_yolov5(results)


    labels = [
        custom_label(model.names[int(class_id)], float(confidence))
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    # Annotate boxes and labels
    annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    out.write(annotated_frame)

    frame_count += 1
    progress = (frame_count / total_frame) * 100
    print(f"\rProcessing: {progress:.2f}% complete", end="")

cap.release()
out.release()
print(f"\nProcessed video saved as {output_path}")
