from ultralytics import YOLO
model = YOLO('./yolo11s.pt')
results = model.train(data="../models/DetectChemistry.yaml",
                      epochs=50,
                      imgsz=640,
                      batch = 32,
                      device='cuda:0',
                      project = "../runs")
