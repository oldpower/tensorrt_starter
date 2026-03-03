from ultralytics import YOLO
import cv2

def val():
    # Validate the model
    metrics = model.val(project="../runs")  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95(B)
    metrics.box.map50  # map50(B)
    metrics.box.map75  # map75(B)
    metrics.box.maps  # a list contains map50-95(B) of each category

model = YOLO("../runs/train/weights/best.pt")  # load an official model
if __name__ == "__main__":
    val()
