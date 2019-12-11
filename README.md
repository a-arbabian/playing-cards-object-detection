# YOLO Object Detection for Playing Cards
Adapting the darknet/YOLO object detection framework for real-time detection of Bicycle playing cards

## Demo
[![Watch the video](https://img.youtube.com/vi/45284ygT4qY/maxresdefault.jpg)](https://youtu.be/45284ygT4qY)

### Results
I ran YOLOv3 on my own generated dataset of 60,000 images for 6,400 iterations (tweaks made to parameters as needed). 
I plan to continue training, but here are the results of the 6400 iter. weights:

| mAP | IoU | F1-Score |
| :--- |:---| :---|
| 99.86% | 82.92% |  0.99 |

### Limitations
* Only tested on Bicycle cards. I chose this deck due to its popularity in commercial use.
