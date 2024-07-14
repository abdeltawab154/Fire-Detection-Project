import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import requests
import threading

class FireDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Fire Detection Application')
        self.root.geometry('900x800')
        self.root.configure(bg='#f0f0f0')

        self.title_frame = tk.Frame(root, bg='#003366', padx=10, pady=10)
        self.title_frame.pack(fill=tk.X)

        self.title_label = tk.Label(self.title_frame, text='Fire Detection Application', fg='white', bg='#003366', font=('Arial', 24, 'bold'))
        self.title_label.pack()

        self.video_frame = tk.Frame(root, bg='#f0f0f0', padx=10, pady=10)
        self.video_frame.pack(pady=10, fill=tk.X)

        self.video_label = tk.Label(self.video_frame, text='Video File Path:', bg='#f0f0f0', font=('Arial', 12))
        self.video_label.pack(pady=5)

        self.video_entry = tk.Entry(self.video_frame, width=60, font=('Arial', 12))
        self.video_entry.pack(pady=5)

        self.video_button = tk.Button(self.video_frame, text="Browse...", command=self.browse_video, width=15, height=2, bg='#007bff', fg='white', font=('Arial', 12), relief=tk.RAISED, borderwidth=2)
        self.video_button.pack(pady=5)

        self.button_frame = tk.Frame(root, bg='#f0f0f0', padx=10, pady=10)
        self.button_frame.pack(pady=10)

        self.start_button = tk.Button(self.button_frame, text="Start Detection", command=self.start_detection, width=20, height=2, bg='#28a745', fg='white', font=('Arial', 14), relief=tk.RAISED, borderwidth=2)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.stop_button = tk.Button(self.button_frame, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED, width=20, height=2, bg='#dc3545', fg='white', font=('Arial', 14), relief=tk.RAISED, borderwidth=2)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.exit_button = tk.Button(self.button_frame, text="Exit", command=self.root.quit, width=20, height=2, bg='#6c757d', fg='white', font=('Arial', 14), relief=tk.RAISED, borderwidth=2)
        self.exit_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.cap = None
        self.detecting = False
        self.video_path = ''
        self.api_key = 'AIzaSyD1UhwP6cGWFZTAlt9LryLT75tnTdr4nGo'  # Hard-coded API Key
        self.database_url = 'https://naser-ba48a-default-rtdb.firebaseio.com/'  # Hard-coded Database URL

        self.FIRE_CLASS_NAME = 'Fire'
        self.FIRE_CLASS_ID = None
        self.model = None

        self.image_label = None  # Initialize the image_label here

    def browse_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if self.video_path:
            self.video_entry.delete(0, tk.END)
            self.video_entry.insert(0, self.video_path)

    def start_detection(self):
        self.video_path = self.video_entry.get()

        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file.")
            return

        try:
            self.model = YOLO('best.pt')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            return

        class_names = self.model.names
        self.FIRE_CLASS_ID = next((i for i, name in class_names.items() if name == self.FIRE_CLASS_NAME), None)

        if self.FIRE_CLASS_ID is None:
            messagebox.showerror("Error", f"'{self.FIRE_CLASS_NAME}' class not found in the model.")
            return

        self.detecting = True
        self.start_button.config(state=tk.DISABLED, bg='#5a6268')
        self.stop_button.config(state=tk.NORMAL, bg='#dc3545')

        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open video.")
            self.detecting = False
            self.start_button.config(state=tk.NORMAL, bg='#28a745')
            self.stop_button.config(state=tk.DISABLED, bg='#dc3545')
            return

        # Start a new thread for processing video frames
        self.thread = threading.Thread(target=self.process_frames)
        self.thread.start()

    def stop_detection(self):
        self.detecting = False
        self.start_button.config(state=tk.NORMAL, bg='#28a745')
        self.stop_button.config(state=tk.DISABLED, bg='#dc3545')

        # Wait for the video processing thread to finish
        if hasattr(self, 'thread'):
            self.thread.join()

        self.update_firebase(0)

    def process_frames(self):
        min_fire_size = 500
        min_aspect_ratio = 0.5
        previous_detection_status = None

        while self.detecting and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to read frame.")
                break

            # Resize frame to improve processing speed
            frame = cv2.resize(frame, (640, 480))

            fire_detected = self.detect_fire(frame, min_fire_size, min_aspect_ratio)

            # Only update Firebase if the detection status has changed
            current_detection_status = 1 if fire_detected else 0
            if current_detection_status != previous_detection_status:
                self.update_firebase(current_detection_status)
                previous_detection_status = current_detection_status
                if fire_detected:
                    print("Fire detected! Motor water pump activated!")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            if self.image_label:
                self.image_label.imgtk = imgtk
                self.image_label.configure(image=imgtk)
            else:
                self.image_label = tk.Label(self.root, image=imgtk)
                self.image_label.pack(pady=10)
                self.image_label.imgtk = imgtk

            self.root.update_idletasks()
            self.root.update()

            # Add a delay to reduce the processing load
            cv2.waitKey(1)

        if self.cap:
            self.cap.release()

    def detect_fire(self, frame, min_fire_size, min_aspect_ratio):
        try:
            results = self.model(frame, conf=0.5)
        except Exception as e:
            print(f"Error: Failed to process frame - {e}")
            return False

        fire_detected = False
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if int(box.cls[0]) == self.FIRE_CLASS_ID:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        aspect_ratio = width / height

                        if area >= min_fire_size and min_aspect_ratio <= aspect_ratio <= (1 / min_aspect_ratio):
                            fire_detected = True
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, 'Fire', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return fire_detected

    def update_firebase(self, value):
        url = f'{self.database_url}/motor_water_pump.json?auth={self.api_key}'
        try:
            response = requests.put(url, json=value)
            if response.status_code != 200:
                print(f"Error updating Firebase: {response.text}")
                messagebox.showerror("Firebase Update Error", "Failed to update Firebase.")
        except Exception as e:
            print(f"Error: Failed to update Firebase - {e}")
            messagebox.showerror("Firebase Update Error", "Failed to update Firebase.")

    def __del__(self):
        if self.cap:
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = FireDetectionApp(root)
    root.mainloop()
