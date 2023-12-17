from PIL import Image
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('C:/Users/sshak/Desktop/Project 3/runs/detect/Good2/weights/best.pt')
# model = YOLO('C:/Users/sshak/Desktop/Project 3/runs/colab/best2.pt')

# Load image and resize it
image = Image.open('C:/Users/sshak/Desktop/Project 3/extracted_motherboard.png')


# Run inference on 'bus.jpg'
results = model(image)  # results list

# Show the results
for r in results:
    im_array = r.plot(pil=True, font_size=15)  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('results3_5.jpg')  # save image