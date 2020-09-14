# FaceMaskDetector
![video detection gif](videoDetection.gif) <br/>

DL4J machine learning model to detect faces wearing and not wearing masks. Model created using transfer learning with YOLOv2. Code features offline validation with static images, video, and laptop webcam

# Pre-requisites to run
1. Clone of this repo

2. IntelliJ. YOLO v2 model automatically downloads when FaceMaskDetection.java is first run. Model downloads
into C:\Users\YourWindowsAccountName\.deeplearning4j\models

3. Images labelled using PascalVOC. Place this in C:\Users\YourWindowsAccountName\.deeplearning4j\data\faceMaskDetector
<br/>
Start up the program from FaceMaskDetection.java in IntelliJ by right-clicking in the view window and clicking "Run"<br/>
![run instruction](runInstruction.jpg .jpg) <br/>

# Data source:
https://www.kaggle.com/andrewmvd/face-mask-detection
