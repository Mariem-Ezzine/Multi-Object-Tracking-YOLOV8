# Multi-Object Tracking – YOLOv8  
A project by **Mariem Ezzine**

## 🚀 Project Overview  
This repository demonstrates a complete pipeline for real-time multi-object tracking (MOT) using the YOLOv8 detector integrated with a tracking algorithm (e.g., DeepSORT, ByteTrack, or SORT). The goal is to detect multiple objects in video streams, maintain unique identities, save trajectories, and support subsequent analysis.  
The project is part of my research interest in computer vision and embedded systems.

## 🎯 Key Features  
- Real-time object detection using YOLOv8  
- Continuous tracking of multiple objects with consistent IDs  
- Visualization of trajectories on video frames  
- Option to output results (videos, bounding boxes + IDs, metrics)  
- Extensible for embedded / industrial use-cases  

## 🧪 Installation & Usage  
1. Clone this repository:  
   ```bash
   git clone https://github.com/Mariem-Ezzine/Multi-Object-Tracking-YOLOv8.git
   cd Multi-Object-Tracking-YOLOv8
Install dependencies:

bash
Copier le code
pip install -r requirements.txt
Run the tracking pipeline (example):

bash
Copier le code
python src/run_tracking.py --source data/samples/my_video.mp4 --output data/outputs/out_video.mp4
Options:

--source : video file path or camera index (e.g., 0)

--output : path for saving annotated output

Additional flags for tracker type or model path can be added

🛠️ Tech Stack
YOLOv8 (Ultralytics) for detection. 
Ultralytics Docs

Tracking algorithm: DeepSORT / ByteTrack / SORT (selectable)

Python, OpenCV for video processing

NumPy for data operations

(Optional) Export to CSV/JSON for trajectories and metrics

📐 Evaluation & Results
Screenshot of output:

Tracking metrics (e.g., ID F1, MOTA) can be computed post-hoc for each run, enabling comparisons of tracker performance.

📚 Context & Applications
Multi-object tracking is a core task in computer vision, used in surveillance, autonomous vehicles, robotics, and industrial inspection. 
ResearchGate
+1

By combining a strong detector (YOLOv8) with a robust tracker, this project targets a high-performance MOT solution for real-time systems.

🔧 Customisation & Extension
You can adapt the project by:

Using your own trained YOLOv8 weights (model=…)

Changing tracker parameters (e.g., IOU threshold, max age)

Adding support for multiple input streams / cameras

Exporting tracking IDs + bounding boxes to a database or UI

📝 Notes
This repository is organized as a demonstration / prototype. For production usage, further work on deployment, edge optimisation, and hardware integration is required.

Training a custom detector is out of scope; the assumption is you either use pretrained weights or fine-tune separately.

👩‍💻 Author
Mariem Ezzine – Electrical Engineer | Computer Vision & Embedded Systems
📍 Tunisia — Open to relocation in Europe
📧 mariemezzine8@gmail.com
🔗 LinkedIn : www.linkedin.com/in/mariem-ezzine
