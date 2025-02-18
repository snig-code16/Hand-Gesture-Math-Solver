import tkinter as tk
from tkinter import ttk, scrolledtext, colorchooser, messagebox, filedialog, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import mediapipe as mp
import threading
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import os
from datetime import datetime
import cvzone
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai

# Set your API key directly in the script (replace with your actual key)
API_KEY = "YOUR_API_KEY_HERE"

# Configure the generative AI model
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

# Colors
PURPLE = (255, 0, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# States
DRAWING = 0
SOLVING = 1
SHOWING_RESULT = 2

def get_hand_info(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

def draw(info, prev_pos, canvas, color=PURPLE, thickness=5):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:  # Index finger down for drawing
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, tuple(prev_pos), tuple(current_pos), color, thickness)
    elif fingers == [1, 0, 0, 0, 0]:  # Thumb up to clear the canvas
        canvas[:] = 0
    return current_pos, canvas

def preprocess_image(canvas):
    # Convert to grayscale and apply thresholding for better AI processing
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    return Image.fromarray(thresh)

def send_to_ai(model, canvas, fingers):
    if fingers == [0, 0, 1, 1, 1]:  # Last 3 fingers down to solve
        pil_image = preprocess_image(canvas)
        prompt = [
            "Analyze the handwritten math problem in this image. Then:",
            "1. Solve the problem step by step.",
            "2. Provide the final numerical answer.",
            "3. Show your work by writing out the full equation and each step of the solution.",
            "Format your response as follows:",
            "ANSWER: [final numerical result]",
            "STEPS:",
            "[Step-by-step solution with equations]"
        ]
        response = model.generate_content(prompt + [pil_image])
        if response and response.text:
            return response.text.strip()
        else:
            return "Error: No response from AI model."
    return None

def parse_ai_response(response):
    lines = response.split('\n')
    answer = "No answer provided"
    steps = []
    current_section = None

    for line in lines:
        if line.startswith("ANSWER:"):
            answer = line.replace("ANSWER:", "").strip()
            current_section = "answer"
        elif line.startswith("STEPS:"):
            current_section = "steps"
        elif current_section == "steps" and line.strip():
            steps.append(line.strip())

    full_equation = "\n".join(steps) if steps else "No detailed steps provided"
    return answer, full_equation

class GestureMathApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GestureMath")
        self.geometry("1600x900")
        self.configure(bg="#1e1e1e")

        self.load_settings()

        self.model = None  # Initialize model later when needed

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=1,
                                         min_detection_confidence=0.7,
                                         min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

        self.canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.prev_pos = None
        self.state = "DRAWING"
        self.ai_result = ""
        self.full_equation = ""
        self.drawing_color = (0, 255, 0)
        self.is_dark_mode = self.settings['dark_mode']
        self.is_paused = False

        self.create_menu()
        self.create_widgets()
        self.update()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_settings(self):
        default_settings = {
            'api_key': 'YOUR_API_KEY_HERE',
            'dark_mode': True,
            'language': 'en'
        }
        try:
            with open('settings.json', 'r') as f:
                self.settings = json.load(f)
        except FileNotFoundError:
            self.settings = default_settings
            self.save_settings()

    def save_settings(self):
        with open('settings.json', 'w') as f:
            json.dump(self.settings, f)

    def create_menu(self):
        self.menu_bar = tk.Menu(self)
        self.config(menu=self.menu_bar)

        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Solution", command=self.save_solution)
        file_menu.add_command(label="Load Solution", command=self.load_solution)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)

        edit_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Preferences", command=self.open_preferences)

        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
        help_menu.add_command(label="About", command=self.show_about)

        features_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Features", menu=features_menu)
        features_menu.add_command(label="Calculator", command=self.open_calculator)
        features_menu.add_command(label="Unit Converter", command=self.open_unit_converter)

    def create_widgets(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_frame = ttk.Frame(self.left_frame)
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas_widget = tk.Canvas(self.video_frame, width=960, height=540, highlightthickness=0)
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.result_label = ttk.Label(self.right_frame, text="Result:", font=("Arial", 16, "bold"))
        self.result_label.pack(pady=10)

        self.result_text = scrolledtext.ScrolledText(self.right_frame, height=15, font=("Arial", 12))
        self.result_text.pack(pady=10, fill=tk.BOTH, expand=True)

        self.graph_frame = ttk.Frame(self.right_frame)
        self.graph_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.gesture_label = ttk.Label(self.right_frame, text="Current Gesture: None", font=("Arial", 14))
        self.gesture_label.pack(pady=10)

        self.create_control_panel()

        self.lighting_label = ttk.Label(self.right_frame, text="Lighting: Checking...", font=("Arial", 12))
        self.lighting_label.pack(pady=10)

        self.status_bar = ttk.Label(self, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.update_theme()

    def create_control_panel(self):
        control_frame = ttk.Frame(self.right_frame)
        control_frame.pack(pady=10, fill=tk.X)

        button_style = ttk.Style()
        button_style.configure('Large.TButton', padding=(20, 10))

        self.color_button = ttk.Button(control_frame, text="Choose Color", command=self.choose_color, style='Large.TButton')
        self.color_button.grid(row=0, column=0, padx=5, pady=5)

        self.clear_button = ttk.Button(control_frame, text="Clear Canvas", command=self.clear_canvas, style='Large.TButton')
        self.clear_button.grid(row=0, column=1, padx=5, pady=5)

        self.solve_button = ttk.Button(control_frame, text="Solve Problem", command=self.solve_problem, style='Large.TButton')
        self.solve_button.grid(row=0, column=2, padx=5, pady=5)

        self.theme_button = ttk.Button(control_frame, text="Toggle Theme", command=self.toggle_theme, style='Large.TButton')
        self.theme_button.grid(row=1, column=0, padx=5, pady=5)

        self.pause_button = ttk.Button(control_frame, text="Pause", command=self.toggle_pause, style='Large.TButton')
        self.pause_button.grid(row=1, column=1, padx=5, pady=5)

        self.manual_input_button = ttk.Button(control_frame, text="Manual Input", command=self.open_manual_input, style='Large.TButton')
        self.manual_input_button.grid(row=1, column=2, padx=5, pady=5)

    def create_calculator_panel(self):
        calc_frame = ttk.Frame(self.right_frame)
        calc_frame.pack(pady=10, fill=tk.X)

        self.calc_display = ttk.Entry(calc_frame, font=("Arial", 14), justify="right")
        self.calc_display.grid(row=0, column=0, columnspan=4, padx=5, pady=5, sticky="nsew")

        buttons = [
            '7', '8', '9', '/',
            '4', '5', '6', '*',
            '1', '2', '3', '-',
            '0', '.', '=', '+'
        ]

        row = 1
        col = 0
        for button in buttons:
            cmd = lambda x=button: self.click_calculator(x)
            ttk.Button(calc_frame, text=button, command=cmd).grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            col += 1
            if col > 3:
                col = 0
                row += 1

        # Add conversion buttons
        ttk.Button(calc_frame, text="Length", command=self.open_length_conversion).grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        ttk.Button(calc_frame, text="Weight", command=self.open_weight_conversion).grid(row=row, column=1, padx=5, pady=5, sticky="nsew")
        ttk.Button(calc_frame, text="Temperature", command=self.open_temperature_conversion).grid(row=row, column=2, padx=5, pady=5, sticky="nsew")
        ttk.Button(calc_frame, text="Currency", command=self.open_currency_conversion).grid(row=row, column=3, padx=5, pady=5, sticky="nsew")

    def click_calculator(self, key):
        if key == '=':
            try:
                result = eval(self.calc_display.get())
                self.calc_display.delete(0, tk.END)
                self.calc_display.insert(tk.END, str(result))
            except:
                self.calc_display.delete(0, tk.END)
                self.calc_display.insert(tk.END, "Error")
        else:
            self.calc_display.insert(tk.END, key)

    def open_length_conversion(self):
        conversion_window = tk.Toplevel(self)
        conversion_window.title("Length Conversion")

        ttk.Label(conversion_window, text="From:").grid(row=0, column=0)
        from_unit = ttk.Combobox(conversion_window, values=["meters", "feet", "inches"])
        from_unit.grid(row=0, column=1)

        ttk.Label(conversion_window, text="To:").grid(row=1, column=0)
        to_unit = ttk.Combobox(conversion_window, values=["meters", "feet", "inches"])
        to_unit.grid(row=1, column=1)

        ttk.Label(conversion_window, text="Value:").grid(row=2, column=0)
        value_entry = ttk.Entry(conversion_window)
        value_entry.grid(row=2, column=1)

        result_label = ttk.Label(conversion_window, text="")
        result_label.grid(row=3, column=0, columnspan=2)

        def convert():
            try:
                value = float(value_entry.get())
                from_unit_val = from_unit.get()
                to_unit_val = to_unit.get()

                # Conversion logic
                if from_unit_val == to_unit_val:
                    result = value
                elif from_unit_val == "meters" and to_unit_val == "feet":
                    result = value * 3.28084
                elif from_unit_val == "meters" and to_unit_val == "inches":
                    result = value * 39.3701
                elif from_unit_val == "feet" and to_unit_val == "meters":
                    result = value / 3.28084
                elif from_unit_val == "feet" and to_unit_val == "inches":
                    result = value * 12
                elif from_unit_val == "inches" and to_unit_val == "meters":
                    result = value / 39.3701
                elif from_unit_val == "inches" and to_unit_val == "feet":
                    result = value / 12
                else:
                    result = "Invalid conversion"

                result_label.config(text=f"Result: {result:.4f} {to_unit_val}")
            except ValueError:
                result_label.config(text="Invalid input")

        ttk.Button(conversion_window, text="Convert", command=convert).grid(row=4, column=0, columnspan=2)

    def open_weight_conversion(self):
        conversion_window = tk.Toplevel(self)
        conversion_window.title("Weight Conversion")

        ttk.Label(conversion_window, text="From:").grid(row=0, column=0)
        from_unit = ttk.Combobox(conversion_window, values=["kg", "lbs", "oz"])
        from_unit.grid(row=0, column=1)

        ttk.Label(conversion_window, text="To:").grid(row=1, column=0)
        to_unit = ttk.Combobox(conversion_window, values=["kg", "lbs", "oz"])
        to_unit.grid(row=1, column=1)

        ttk.Label(conversion_window, text="Value:").grid(row=2, column=0)
        value_entry = ttk.Entry(conversion_window)
        value_entry.grid(row=2, column=1)

        result_label = ttk.Label(conversion_window, text="")
        result_label.grid(row=3, column=0, columnspan=2)

        def convert():
            try:
                value = float(value_entry.get())
                from_unit_val = from_unit.get()
                to_unit_val = to_unit.get()

                # Conversion logic
                if from_unit_val == to_unit_val:
                    result = value
                elif from_unit_val == "kg" and to_unit_val == "lbs":
                    result = value * 2.20462
                elif from_unit_val == "kg" and to_unit_val == "oz":
                    result = value * 35.274
                elif from_unit_val == "lbs" and to_unit_val == "kg":
                    result = value / 2.20462
                elif from_unit_val == "lbs" and to_unit_val == "oz":
                    result = value * 16
                elif from_unit_val == "oz" and to_unit_val == "kg":
                    result = value / 35.274
                elif from_unit_val == "oz" and to_unit_val == "lbs":
                    result = value / 16
                else:
                    result = "Invalid conversion"

                result_label.config(text=f"Result: {result:.4f} {to_unit_val}")
            except ValueError:
                result_label.config(text="Invalid input")

        ttk.Button(conversion_window, text="Convert", command=convert).grid(row=4, column=0, columnspan=2)

    def open_temperature_conversion(self):
        conversion_window = tk.Toplevel(self)
        conversion_window.title("Temperature Conversion")

        ttk.Label(conversion_window, text="From:").grid(row=0, column=0)
        from_unit = ttk.Combobox(conversion_window, values=["Celsius", "Fahrenheit", "Kelvin"])
        from_unit.grid(row=0, column=1)

        ttk.Label(conversion_window, text="To:").grid(row=1, column=0)
        to_unit = ttk.Combobox(conversion_window, values=["Celsius", "Fahrenheit", "Kelvin"])
        to_unit.grid(row=1, column=1)

        ttk.Label(conversion_window, text="Value:").grid(row=2, column=0)
        value_entry = ttk.Entry(conversion_window)
        value_entry.grid(row=2, column=1)

        result_label = ttk.Label(conversion_window, text="")
        result_label.grid(row=3, column=0, columnspan=2)

        def convert():
            try:
                value = float(value_entry.get())
                from_unit_val = from_unit.get()
                to_unit_val = to_unit.get()

                # Conversion logic
                if from_unit_val == to_unit_val:
                    result = value
                elif from_unit_val == "Celsius" and to_unit_val == "Fahrenheit":
                    result = (value * 9/5) + 32
                elif from_unit_val == "Celsius" and to_unit_val == "Kelvin":
                    result = value + 273.15
                elif from_unit_val == "Fahrenheit" and to_unit_val == "Celsius":
                    result = (value - 32) * 5/9
                elif from_unit_val == "Fahrenheit" and to_unit_val == "Kelvin":
                    result = (value - 32) * 5/9 + 273.15
                elif from_unit_val == "Kelvin" and to_unit_val == "Celsius":
                    result = value - 273.15
                elif from_unit_val == "Kelvin" and to_unit_val == "Fahrenheit":
                    result = (value - 273.15) * 9/5 + 32
                else:
                    result = "Invalid conversion"

                result_label.config(text=f"Result: {result:.2f} {to_unit_val}")
            except ValueError:
                result_label.config(text="Invalid input")

        ttk.Button(conversion_window, text="Convert", command=convert).grid(row=4, column=0, columnspan=2)

    def open_currency_conversion(self):
        conversion_window = tk.Toplevel(self)
        conversion_window.title("Currency Conversion")

        ttk.Label(conversion_window, text="From:").grid(row=0, column=0)
        from_currency = ttk.Combobox(conversion_window, values=["USD", "EUR", "GBP", "JPY"])
        from_currency.grid(row=0, column=1)

        ttk.Label(conversion_window, text="To:").grid(row=1, column=0)
        to_currency = ttk.Combobox(conversion_window, values=["USD", "EUR", "GBP", "JPY"])
        to_currency.grid(row=1, column=1)

        ttk.Label(conversion_window, text="Amount:").grid(row=2, column=0)
        amount_entry = ttk.Entry(conversion_window)
        amount_entry.grid(row=2, column=1)

        result_label = ttk.Label(conversion_window, text="")
        result_label.grid(row=3, column=0, columnspan=2)

        def convert():
            try:
                amount = float(amount_entry.get())
                from_curr = from_currency.get()
                to_curr = to_currency.get()

                # Note: These are example exchange rates and should be updated regularly in a real application
                rates = {
                    "USD": 1.0,
                    "EUR": 0.85,
                    "GBP": 0.73,
                    "JPY": 110.0
                }

                if from_curr in rates and to_curr in rates:
                    result = amount * (rates[to_curr] / rates[from_curr])
                    result_label.config(text=f"Result: {result:.2f} {to_curr}")
                else:
                    result_label.config(text="Invalid currency selection")
            except ValueError:
                result_label.config(text="Invalid input")

        ttk.Button(conversion_window, text="Convert", command=convert).grid(row=4, column=0, columnspan=2)

    def update(self):
        if not self.is_paused:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                self.check_lighting(frame)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        self.handle_gestures(hand_landmarks, frame)
                else:
                    self.gesture_label.config(text="Current Gesture: None")
                    self.prev_pos = None

                mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                frame_bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
                frame_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
                combined_image = cv2.add(frame_bg, frame_fg)

                self.photo = self.convert_to_photo(combined_image)
                self.canvas_widget.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.after(10, self.update)

    def handle_gestures(self, hand_landmarks, frame):
        image_height, image_width, _ = frame.shape
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        # Convert normalized coordinates to pixel coordinates
        thumb_x, thumb_y = int(thumb_tip.x * image_width), int(thumb_tip.y * image_height)
        index_x, index_y = int(index_tip.x * image_width), int(index_tip.y * image_height)
        middle_x, middle_y = int(middle_tip.x * image_width), int(middle_tip.y * image_height)
        ring_x, ring_y = int(ring_tip.x * image_width), int(ring_tip.y * image_height)
        pinky_x, pinky_y = int(pinky_tip.x * image_width), int(pinky_tip.y * image_height)

        # Define gesture thresholds
        drawing_threshold = 50
        thumb_up_threshold = image_height * 0.3

        if thumb_y < thumb_up_threshold:
            self.gesture_label.config(text="Current Gesture: Thumb Up (Clear Canvas)")
            self.clear_canvas()
        elif abs(index_y - middle_y) < drawing_threshold and index_y < ring_y and index_y < pinky_y:
            self.gesture_label.config(text="Current Gesture: Drawing")
            if self.prev_pos:
                cv2.line(self.canvas, self.prev_pos, (index_x, index_y), self.drawing_color, 5)
            self.prev_pos = (index_x, index_y)
        elif index_y < middle_y and middle_y < ring_y and ring_y < pinky_y:
            self.gesture_label.config(text="Current Gesture: Solve Problem")
            self.solve_problem()
        else:
            self.gesture_label.config(text="Current Gesture: None")
            self.prev_pos = None

    def check_lighting(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        average_brightness = np.mean(gray_frame)
        if average_brightness < 100:
            self.lighting_label.config(text="Lighting: Poor", foreground="red")
        elif average_brightness < 150:
            self.lighting_label.config(text="Lighting: Moderate", foreground="yellow")
        else:
            self.lighting_label.config(text="Lighting: Good", foreground="green")

    def convert_to_photo(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(image=Image.fromarray(rgb_image))

    def choose_color(self):
        color_code = colorchooser.askcolor(title="Choose drawing color")
        if color_code[1]:
            self.drawing_color = tuple(int(color_code[0][i]) for i in range(3))

    def clear_canvas(self):
        self.canvas.fill(0)
        self.prev_pos = None
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Canvas cleared.\n")

        for widget in self.graph_frame.winfo_children():
            widget.destroy()

    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        self.settings['dark_mode'] = self.is_dark_mode
        self.save_settings()
        self.update_theme()

    def update_theme(self):
        if self.is_dark_mode:
            self.style.configure("TFrame", background="#1e1e1e")
            self.style.configure("TLabel", background="#1e1e1e", foreground="#ffffff")
            self.style.configure("TButton", background="#4a4a4a", foreground="#ffffff")
            self.result_text.configure(bg="#2a2a2a", fg="#ffffff")
            self.canvas_widget.configure(bg="#2a2a2a")
        else:
            self.style.configure("TFrame", background="#f0f0f0")
            self.style.configure("TLabel", background="#f0f0f0", foreground="#000000")
            self.style.configure("TButton", background="#e0e0e0", foreground="#000000")
            self.result_text.configure(bg="#ffffff", fg="#000000")
            self.canvas_widget.configure(bg="#ffffff")

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_button.config(text="Resume")
        else:
            self.pause_button.config(text="Pause")

    def open_manual_input(self):
        input_window = tk.Toplevel(self)
        input_window.title("Manual Input")
        input_window.geometry("400x300")

        input_label = ttk.Label(input_window, text="Enter your math problem:")
        input_label.pack(pady=10)

        input_text = scrolledtext.ScrolledText(input_window, height=10, width=40)
        input_text.pack(pady=10)

        submit_button = ttk.Button(input_window, text="Solve", command=lambda: self.solve_manual_input(input_text.get("1.0", tk.END).strip()))
        submit_button.pack(pady=10)

    def solve_manual_input(self, problem):
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Solving manually entered problem...\n")

        prompt = [
            f"Solve the following math problem:\n{problem}\n",
            "Please provide a step-by-step solution, explaining each step clearly.",
            "Format your response as plain text, without using markdown or special formatting.",
            "Include the problem type, solution steps, and final answer."
        ]

        try:
            response = self.model.generate_content(prompt)
            self.ai_result = response.text

            self.ai_result = self.ai_result.replace('#', '').replace('', '')

            self.result_text.insert(tk.END, self.ai_result)
            self.plot_graph_if_available()
        except Exception as e:
            self.result_text.insert(tk.END, f"An error occurred: {str(e)}")

    def solve_problem(self):
        threading.Thread(target=self._solve_problem_thread).start()

    def _solve_problem_thread(self):
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Solving...\n")
        self.status_bar.config(text="Solving problem...")
        start_time = datetime.now()

        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        pil_image = Image.fromarray(thresh)

        prompt = [
            "Analyze the handwritten math problem in this image. Provide a step-by-step solution, explaining each step clearly. Format your response as plain text, without using markdown or special formatting. Include the problem type, solution steps, and final answer."
        ]

        encoded_image = self.image_to_base64(pil_image)

        try:
            response = self.model.generate_content([prompt, encoded_image])
            self.ai_result = response.text

            self.ai_result = self.ai_result.replace('#', '').replace('', '')

            self.result_text.insert(tk.END, self.ai_result)
            self.plot_graph_if_available()

            end_time = datetime.now()
            solving_time = (end_time - start_time).total_seconds()
            self.status_bar.config(text=f"Problem solved in {solving_time:.2f} seconds")
        except Exception as e:
            self.result_text.insert(tk.END, f"An error occurred: {str(e)}")
            self.status_bar.config(text="Error occurred while solving problem")

    def plot_graph_if_available(self):
        if "GRAPH:" in self.ai_result:
            graph_data = self.ai_result.split("GRAPH:")[1].split("KEY FEATURES:")[0].strip()

            fig, ax = plt.subplots(figsize=(5, 4))
            # Parse the graph data and plot it accordingly
            # ... (implementation based on the specific format of graph_data)
            ax.set_title("Graph Representation")
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")

            canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)

            canvas.draw()

    def image_to_base64(self, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def save_solution(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", ".txt"), ("All files", ".*")])
        if file_path:
            with open(file_path, 'w') as file:
                file.write(self.result_text.get(1.0, tk.END))
            messagebox.showinfo("Save Solution", "Solution saved successfully!")

    def load_solution(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", ".txt"), ("All files", ".*")])
        if file_path:
            with open(file_path, 'r') as file:
                content = file.read()
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, content)

    def open_preferences(self):
        pref_window = tk.Toplevel(self)
        pref_window.title("Preferences")
        pref_window.geometry("300x200")

        ttk.Label(pref_window, text="API Key:").pack(pady=5)
        api_key_entry = ttk.Entry(pref_window, width=40)
        api_key_entry.insert(0, self.settings['api_key'])
        api_key_entry.pack(pady=5)

        ttk.Label(pref_window, text="Language:").pack(pady=5)
        lang_var = tk.StringVar(value=self.settings['language'])
        lang_combobox = ttk.Combobox(pref_window, textvariable=lang_var, values=['en', 'es', 'fr', 'de', 'it'])
        lang_combobox.pack(pady=5)

        def save_preferences():
            self.settings['api_key'] = api_key_entry.get()
            self.settings['language'] = lang_var.get()
            self.save_settings()
            self.model = genai.GenerativeModel('gemini-1.5-flash')  # Initialize model with updated API key
            pref_window.destroy()

        ttk.Button(pref_window, text="Save", command=save_preferences).pack(pady=10)

    def show_user_guide(self):
        guide = """
        GestureMath User Guide:
        1. Use your index finger to draw on the canvas.
        2. Use two fingers (index and middle) to stop drawing.
        3. Raise your thumb to clear the canvas.
        4. Raise your last three fingers to solve the problem.
        5. Use the control panel buttons for additional functions.
        6. You can manually input problems using the 'Manual Input' button.
        7. Save and load solutions using the File menu.
        8. Use the built-in calculator for quick calculations.
        9. Perform unit conversions using the conversion buttons.
        10. Customize your experience in the Preferences window.
        """
        messagebox.showinfo("User Guide", guide)

    def show_about(self):
        about_text = """
        GestureMath v2.0
        Developed by Team codeARC
        Members: Chirag Nahata, Snigdha Ghosh, Srijita Saha

        © 2024 Team codeARC. All rights reserved.

        An AI-powered math solver with gesture recognition and advanced mathematical tools.
        """
        messagebox.showinfo("About GestureMath", about_text)

    def open_calculator(self):
        calculator_window = tk.Toplevel(self)
        calculator_window.title("Calculator")
        calculator_window.geometry("300x300")

        # Create calculator interface here

    def open_unit_converter(self):
        converter_window = tk.Toplevel(self)
        converter_window.title("Unit Converter")
        converter_window.geometry("300x300")

        # Create unit converter interface here

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.cap.release()
            self.destroy()

if __name__ == "__main__":
    app = GestureMathApp()
    app.mainloop()