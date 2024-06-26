# YOLO Object Detection for Playing Cards
Training a YOLOv2 model with the Darknet object detection framework for real-time detection of playing cards

## Demo
Click the video below:
[![Watch the video](https://img.youtube.com/vi/45284ygT4qY/maxresdefault.jpg)](https://youtu.be/45284ygT4qY)

### Results
I ran YOLOv3 on my own generated dataset of 60,000 images (20% saved as test set). 
Results:
| mAP | IoU | F1-Score |
| :--- |:---| :---|
| 99.86% | 82.92% |  0.99 |

The script runs inference at real-time (30fps) on a live video feed.
### Limitations
* Only tested on Bicycle cards. I chose this deck due to its popularity in commercial use.
  
