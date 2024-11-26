from ultralytics import YOLO
import cv2


img = cv2.imread('OD_Assets/bus_students.jpg')
img = cv2.resize(img, (600,400))


# creating the model

# Using default version

# There are different types of weights, the nano, the medium, the large.
# Nano will bw faster, Medium will be a bit slower and Large will be slower.
# It's based upon us to decide which to use. 

classNames = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


# NANO
model = YOLO('YOLO_Weights/yolov8n.pt')           # state the weights needed and it will download              yolo version 8 n for nano
# 448x640 5 persons, 1 bus, 1 backpack, 1 handbag, 121.5ms Speed: 8.7ms preprocess, 121.5ms inference, 0.0ms postprocess per image at shape (1, 3, 448, 640)

# MEDIUM
#model = YOLO('YOLO_Weights/yolov8m.pt')           # 448x640 5 persons, 1 bus, 5 backpacks, 1 handbag, 1 book, 465.9ms Speed: 3.6ms preprocess, 465.9ms inference, 4.3ms postprocess per image at shape (1, 3, 448, 640)

# LARGE
#model = YOLO('YOLO_Weights/yolov8l.pt')            # 448x640 5 persons, 1 bus, 5 backpacks, 1 handbag, 2 books, 910.4ms Speed: 8.0ms preprocess, 910.4ms inference, 0.0ms postprocess per image at shape (1, 3, 448, 640)

#results = model(img, show=True)

#results = model('OD_Assets/cars.jpg', show=True)    #  448x640 20 cars, 121.8ms Speed: 7.0ms preprocess, 121.8ms inference, 0.0ms postprocess per image at shape (1, 3, 448, 640)


# The closer is the object to the camera the better it is detected.
# Basically the idea is how clear is the site of viewing.


cv2.waitKey(0)



#  ---  YOLO WITH WEBCAM  ---


import cvzone                                       # used to display all the dtections in a little bit easier manner
import math


# setting up webcam 

#cap = cv2.VideoCapture(0)                           # for webcam
#cap.set(3, 640)
#cap.set(4, 480)

cap = cv2.VideoCapture('OD_Assets/people.mp4')                          # for videos

while True :
    success, frame = cap.read()

    #frame = cv2.resize(frame, (640, 480))           # resizing the video file 

    results = model(frame, stream=True)             # stream=True use generators which is more efficient 
    
    for result in results :
        boxes = result.boxes
        for box in boxes :
            # OPENCV
            
            x1, y1, x2, y2 = box.xyxy[0]
            #print(x1, y1, x2, y2)

            # converting x1, y1, x2, y2 values into intergesrs in order to read them
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
            #print(x1, y1, x2, y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
           
            # CVZONE

            #w, h = x2-x1, y2-y1
            #cvzone.cornerRect(frame, (x1, y1, w, h))
            
            # finding confidence
            conf = math.ceil((box.conf[0]*100))/100
            #print(conf)

            # displaying confidence and class name            
            cls = int(box.cls[0])
            cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=2)
    
    cv2.imshow("Image", frame)
            
    if cv2.waitKey(1) & 0xFF == ord('x') :
        break;


# This is how webcam is used with the YOLOv8 to detect different objects.
# Using it with CPU, (that's why the large version is very slow). Quite inefficient!



#  --- YOLO WITH GPU  ---

# Install CUDA Toolkit and CUDNN Toolkit  --> copy CUDNN files into CUDA  --->  ensure the path is correct in environment variables  --> Restart
# CUDA 11.8 and CUDNN 8.6.0 for CUDA 11.x
# Install compatible torch and torchvision     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


