"""
webcam_test.py
--------------
Opens the webcam and displays a live feed in a window.
Press Q to quit.
"""

import cv2

def main():
    # Open the default webcam (index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check that your camera is connected and not in use.")
        return

    print("Webcam opened successfully. Press Q to close the window.")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Failed to grab frame from webcam.")
            break

        # Display a label on the frame
        cv2.putText(
            frame,
            "Virtual Sign Language Tutor - Press Q to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Show the live feed
        cv2.imshow("Webcam Feed", frame)

        # Break loop when Q key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Q pressed — closing webcam.")
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
