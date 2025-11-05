import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog, simpledialog
from boxmot.trackers.bytetrack.byte_tracker import BYTETracker
import torch

class CompleteVideoTracker:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")
        self.root = tk.Tk()
        self.root.withdraw()
        self.max_frames = 134 
        # Demander à l'utilisateur de choisir une vidéo
        self.video_path = filedialog.askopenfilename(
            title="Sélectionnez la vidéo à analyser",
            filetypes=[("Fichiers vidéo", "*.webm *.mp4 *.avi *.mov"), ("Tous les fichiers", "*.*")]
        )
    
        if not self.video_path:
            print("Aucune vidéo sélectionnée. Arrêt du programme.")
            exit()
        
        # Demander le nom des résultats
        self.experiment_name = simpledialog.askstring(
            "Nom des résultats", 
            "Entrez un nom pour cette analyse (sans espaces):",
            parent=self.root
        )
        if not self.experiment_name:
            self.experiment_name = "resultats_tracking"
    
        # Initialiser le modèle YOLO pour la détection
        self.model = YOLO('yolov8n.pt').to(self.device)
    
        # Initialiser ByteTrack avec des paramètres plus permissifs
        self.tracker = BYTETracker(
            track_thresh=0.15,  # Lower threshold to track more objects
            match_thresh=0.7,   # More permissive matching
            frame_rate=30,
            track_buffer=60      # Longer buffer to keep tracks alive
        )
    
        self.cap = cv2.VideoCapture(self.video_path)
    
        # Configurer les dossiers de sortie
        self.setup_output_directory()
    
        # Initialiser les métriques
        self.initialize_metrics()
    
        # Initialiser le writer vidéo
        self.setup_video_writer()
        
    def setup_output_directory(self):
        """Crée un dossier organisé pour les résultats"""
        self.output_dir = f"tracking_results_{self.experiment_name}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Sous-dossiers
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.videos_dir = os.path.join(self.output_dir, "videos")
        self.reports_dir = os.path.join(self.output_dir, "reports")
        
        for folder in [self.plots_dir, self.videos_dir, self.reports_dir]:
            os.makedirs(folder, exist_ok=True)
            
    def setup_video_writer(self):
        """Initialise le writer pour la vidéo de sortie"""
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.output_video_path = os.path.join(
            self.videos_dir, 
            f"tracking_output_{self.experiment_name}.mp4"
        )
        
        self.video_writer = cv2.VideoWriter(
            self.output_video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )
            
    def initialize_metrics(self):
        """Initialise les structures de données pour les métriques"""
        self.metrics = {
            'iou': [],
            'mota': [],
            'idf1': [],
            'frame_count': [],
            'object_count': []  # New metric to track number of objects
        }
        
        # Compteurs pour les métriques globales
        self.gt_count = 0
        self.misses = 0
        self.false_positives = 0
        self.id_switches = 0
        self.id_matches = 0
        self.id_fp = 0
        self.id_fn = 0
        
    def calculate_iou(self, box1, box2):
        """Calcule l'IoU entre deux boîtes"""
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - intersection
        
        return intersection / union if union > 0 else 0
    
    def update_metrics(self, frame_idx, boxes, track_ids):
        """Met à jour toutes les métriques"""
        # Simulation de ground truth (à remplacer par vos vraies données si disponibles)
        gt_boxes = [box + np.random.normal(0, 5, 4) for box in boxes]
        
        # Calcul IoU
        ious = [self.calculate_iou(box, gt_box) for box, gt_box in zip(boxes, gt_boxes)]
        if ious:
            self.metrics['iou'].append(np.mean(ious))
            self.metrics['frame_count'].append(frame_idx)
            self.metrics['object_count'].append(len(boxes))  # Track number of objects
        
        # Calcul MOTA (version simplifiée)
        num_gt = max(len(boxes), 3)  # Simulation
        self.gt_count += num_gt
        self.misses += max(0, num_gt - len(boxes))
        self.false_positives += max(0, len(boxes) - num_gt)
        
        # Simulation d'ID switches (pourrait être amélioré avec un vrai suivi)
        if hasattr(self, 'prev_track_ids'):
            common_ids = set(track_ids) & set(self.prev_track_ids)
            self.id_switches += max(0, len(track_ids) - len(common_ids))
        
        self.prev_track_ids = track_ids
        
        mota = 1 - (self.misses + self.false_positives + self.id_switches) / max(1, self.gt_count)
        self.metrics['mota'].append(mota)
        
        # Calcul IDF1 (version simplifiée)
        matches = min(len(boxes), num_gt)
        self.id_matches += matches
        self.id_fp += max(0, len(boxes) - num_gt)
        self.id_fn += max(0, num_gt - len(boxes))
        
        precision = self.id_matches / (self.id_matches + self.id_fp) if (self.id_matches + self.id_fp) > 0 else 0
        recall = self.id_matches / (self.id_matches + self.id_fn) if (self.id_matches + self.id_fn) > 0 else 0
        idf1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        self.metrics['idf1'].append(idf1)
    
    def process_video(self):
        """Traite la vidéo frame par frame"""
        frame_idx = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame_idx >= self.max_frames:
                break
                
            frame_idx += 1
            display_frame = frame.copy()
    
            # Détection avec YOLO - détecter toutes les classes
            results = self.model(frame, classes=list(range(80)))  # Detect all 80 COCO classes
            
            detections = []
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
                for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                    # Seuil de confiance plus bas pour détecter plus d'objets
                    if conf < 0.3:  # Lower confidence threshold
                        continue
                        
                    x1, y1, x2, y2 = map(int, box)
                    detections.append([x1, y1, x2, y2, conf, cls_id])
            
            # Convertir en format numpy pour ByteTrack
            if detections:
                detections = np.array(detections)
            else:
                detections = np.empty((0, 6))
            
            # Mise à jour du tracker avec toutes les détections
            tracks = self.tracker.update(detections, frame)
            
            # Afficher les résultats
            boxes_list = []
            track_ids = []
            
            for track in tracks:
                track_id = int(track[4])
                x1, y1, x2, y2 = map(int, track[:4])
                
                boxes_list.append([x1, y1, x2, y2])
                track_ids.append(track_id)
                
                # Dessiner la boîte et l'ID
                color = (0, 255, 0)  # Vert pour les tracks valides
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, f"ID : {track_id}", (x1, y1-10),  # Format "ID : X"
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
            # Mise à jour des métriques
            if len(boxes_list) > 0:
                boxes_array = np.array(boxes_list)
                self.update_metrics(frame_idx, boxes_array, track_ids)
    
            # Affichage des métriques
            if self.metrics['iou']:
                cv2.putText(display_frame, f"IoU: {self.metrics['iou'][-1]:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if self.metrics['mota']:
                cv2.putText(display_frame, f"MOTA: {self.metrics['mota'][-1]:.2f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if self.metrics['idf1']:
                cv2.putText(display_frame, f"IDF1: {self.metrics['idf1'][-1]:.2f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if self.metrics['object_count']:
                cv2.putText(display_frame, f"Objects: {self.metrics['object_count'][-1]}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
            # Affichage & enregistrement
            cv2.imshow("Tracking Results", display_frame)
            self.video_writer.write(display_frame)
    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        self.video_writer.release()
        cv2.destroyAllWindows()

        self.generate_plots()
        self.generate_report()

    def generate_plots(self):
        """Génère et sauvegarde les courbes des métriques"""
        plt.figure(figsize=(12, 10))
        
        # Courbe IoU
        plt.subplot(4, 1, 1)
        plt.plot(self.metrics['frame_count'], self.metrics['iou'], 'b-')
        plt.title('Evolution du IoU moyen par frame')
        plt.ylabel('IoU')
        plt.grid(True)
        
        # Courbe MOTA
        plt.subplot(4, 1, 2)
        plt.plot(self.metrics['frame_count'], self.metrics['mota'], 'g-')
        plt.title('Evolution du MOTA par frame')
        plt.ylabel('MOTA')
        plt.grid(True)
        
        # Courbe IDF1
        plt.subplot(4, 1, 3)
        plt.plot(self.metrics['frame_count'], self.metrics['idf1'], 'r-')
        plt.title('Evolution du IDF1 par frame')
        plt.ylabel('IDF1')
        plt.grid(True)
        
        # Courbe nombre d'objets
        plt.subplot(4, 1, 4)
        plt.plot(self.metrics['frame_count'], self.metrics['object_count'], 'm-')
        plt.title('Nombre d\'objets trackés par frame')
        plt.xlabel('Numéro de frame')
        plt.ylabel('Objets')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'all_metrics.png'))
        plt.close()
        
        # Courbe combinée
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['frame_count'], self.metrics['iou'], 'b-', label='IoU')
        plt.plot(self.metrics['frame_count'], self.metrics['mota'], 'g-', label='MOTA')
        plt.plot(self.metrics['frame_count'], self.metrics['idf1'], 'r-', label='IDF1')
        plt.plot(self.metrics['frame_count'], self.metrics['object_count'], 'm-', label='Objects')
        plt.title('Comparaison des métriques de tracking')
        plt.xlabel('Numéro de frame')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, 'combined_metrics.png'))
        plt.close()
    
    def generate_report(self):
        """Génère un rapport textuel détaillé"""
        report_path = os.path.join(self.reports_dir, f'report_{self.experiment_name}.txt')
        
        with open(report_path, 'w') as f:
            f.write(f"Rapport d'analyse - {self.experiment_name}\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Vidéo analysée: {os.path.basename(self.video_path)}\n")
            f.write(f"Vidéo de sortie: tracking_output_{self.experiment_name}.mp4\n")
            f.write(f"Nombre total de frames: {self.metrics['frame_count'][-1] if self.metrics['frame_count'] else 0}\n\n")
            
            f.write("Résultats globaux:\n")
            f.write("-"*50 + "\n")
            
            if self.metrics['iou']:
                avg_iou = np.mean(self.metrics['iou'])
                f.write(f"IoU moyen: {avg_iou:.3f}\n")
            
            if self.metrics['mota']:
                final_mota = self.metrics['mota'][-1]
                f.write(f"MOTA final: {final_mota:.3f}\n")
            
            if self.metrics['idf1']:
                final_idf1 = self.metrics['idf1'][-1]
                f.write(f"IDF1 final: {final_idf1:.3f}\n")
                
            if self.metrics['object_count']:
                avg_objects = np.mean(self.metrics['object_count'])
                f.write(f"Nombre moyen d'objets trackés: {avg_objects:.1f}\n\n")
            
            f.write("Détails par frame:\n")
            f.write("-"*50 + "\n")
            f.write("Frame\tIoU\tMOTA\tIDF1\tObjects\n")
            f.write("-----\t---\t----\t----\t-------\n")
            
            for i, frame in enumerate(self.metrics['frame_count']):
                iou = self.metrics['iou'][i] if i < len(self.metrics['iou']) else 0
                mota = self.metrics['mota'][i] if i < len(self.metrics['mota']) else 0
                idf1 = self.metrics['idf1'][i] if i < len(self.metrics['idf1']) else 0
                objects = self.metrics['object_count'][i] if i < len(self.metrics['object_count']) else 0
                f.write(f"{frame}\t{iou:.3f}\t{mota:.3f}\t{idf1:.3f}\t{objects}\n")

if __name__ == "__main__":
    print("=== Tracking avec ByteTrack et métriques complètes ===")
    print("Sélectionnez votre vidéo et donnez un nom à l'analyse...")
    tracker = CompleteVideoTracker()
    tracker.process_video()