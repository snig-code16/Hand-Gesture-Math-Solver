import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import time

# Set your API key directly in the script
API_KEY = "AIzaSyDcMZQj-kDOyG35D86mT3zJPb_4OO0I7dE"

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
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, tuple(prev_pos), tuple(current_pos), color, thickness)
    elif fingers == [1, 0, 0, 0, 0]:
        canvas[:] = 0  # Clear the canvas
    return current_pos, canvas

def preprocess_image(canvas):
    # Convert to grayscale
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to make the drawing clearer
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    return Image.fromarray(thresh)

def send_to_ai(model, canvas, fingers):
    if fingers == [0, 0, 1, 1, 1]:
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

def main():
    prev_pos = None
    canvas = None
    ai_result = ""
    full_equation = ""
    state = DRAWING
    start_time = time.time()
    display_text = "Draw your math problem here"

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        if canvas is None:
            canvas = np.zeros_like(img)
        
        info = get_hand_info(img)
        if info:
            fingers, lmList = info
            if state == DRAWING:
                prev_pos, canvas = draw(info, prev_pos, canvas)
                if fingers == [0, 0, 1, 1, 1]:
                    state = SOLVING
                    start_time = time.time()
            elif state == SOLVING:
                result = send_to_ai(model, canvas, fingers)
                if result:
                    ai_result, full_equation = parse_ai_response(result)
                    state = SHOWING_RESULT
            
            if fingers == [1, 0, 0, 0, 0]:
                canvas[:] = 0  # Clear the canvas
                ai_result = ""
                full_equation = ""
                state = DRAWING

        image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)

        # Display interactive and feedback text
        if state == DRAWING:
            cv2.putText(image_combined, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2, cv2.LINE_AA)
        elif state == SOLVING:
            elapsed_time = time.time() - start_time
            cv2.putText(image_combined, f"Solving... {elapsed_time:.1f}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2, cv2.LINE_AA)
        elif state == SHOWING_RESULT:
            cv2.putText(image_combined, f"Answer: {ai_result}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2, cv2.LINE_AA)
            y_offset = 100
            for line in full_equation.split('\n'):
                cv2.putText(image_combined, line, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
                y_offset += 30

        # Display instructions
        cv2.putText(image_combined, "Gestures:", (50, image_combined.shape[0] - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
        cv2.putText(image_combined, "- Index finger: Draw", (70, image_combined.shape[0] - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
        cv2.putText(image_combined, "- Thumb: Clear canvas", (70, image_combined.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
        cv2.putText(image_combined, "- Last 3 fingers: Solve", (70, image_combined.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)

        cv2.imshow("Math Problem Solver", image_combined)
        
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()