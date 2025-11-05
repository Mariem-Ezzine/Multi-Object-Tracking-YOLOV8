# ⚡ YOLOv8 + BYTETrack Multi-Object Tracking

Real-time multi-object tracking based on:
- YOLOv8 detector
- BYTETrack tracker (high speed & low latency)
- GPU support when available

✅ Optimized detection thresholds  
✅ Robust tracking of small & fast moving objects  
✅ Live evaluation: IoU, MOTA, IDF1, Object count  
✅ Annotated video + plots + metric report  

---

### ▶️ Run
```bash
python byte_tracker.py

You will select:
✅ Input video
✅ Experiment name

🧪 Metrics (Simulated Ground Truth)

IoU

MOTA

IDF1

Number of tracked objects

Metrics are for demonstration purposes only (no true GT).

📁 Output Structure

Inside: tracking_results_<experiment>/

└── videos/
└── plots/
└── reports/

✅ Future Enhancements

Benchmark on MOT datasets

Real ground truth evaluation

Edge AI deployment

📬 Email: mariemezzine8@gmail.com
