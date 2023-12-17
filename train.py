from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

if __name__ == '__main__':
   # Train Model
   results = model.train(data='C:/Users/sshak/Desktop/Project 3/data/data.yaml', epochs=150, batch=4, imgsz=1000 ,name='cb_yolov8n_v')

