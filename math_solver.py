import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image

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

def get_hand_info(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, tuple(current_pos), tuple(prev_pos), (255, 0, 255), 10)
    elif fingers == [1, 0, 0, 0, 0]:
        canvas[:] = 0  # Clear the canvas
    return current_pos, canvas

def send_to_ai(model, canvas, fingers):
    if fingers == [0, 0, 1, 1, 1]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem", pil_image])
        if response and response.text:
            return response.text.strip()  # Ensure no leading/trailing whitespace
        else:
            return "Error: No response from AI model."
    return None

def main():
    prev_pos = None
    canvas = None
    ai_result = ""
    display_text = "Draw the math problem here"

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        if canvas is None:
            canvas = np.zeros_like(img)
        
        info = get_hand_info(img)
        if info:
            fingers, lmList = info
            prev_pos, canvas = draw(info, prev_pos, canvas)
            result = send_to_ai(model, canvas, fingers)
            if result:
                ai_result = result
            elif fingers == [1, 0, 0, 0, 0]:
                ai_result = ""  # Clear AI result when canvas is cleared

        image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)

        # Clear previous text
        image_combined[:] = cv2.addWeighted(image_combined, 1, np.zeros_like(image_combined), 0, 0)

        # Display interactive and feedback text
        if ai_result:
            cv2.putText(image_combined, "Math problem is:", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image_combined, ai_result, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(image_combined, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Image Combined", image_combined)
        
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()