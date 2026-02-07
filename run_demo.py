import cv2
import json
import numpy as np
from src.non_verbal_analysis.analyzer import NonVerbalAnalyzer

def run_realtime_test():
    # Initialize the module
    analyzer = NonVerbalAnalyzer()
    session_id = "live_dev_test"
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("--- Non-Verbal Analysis Module Live Test ---")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame through the module
        # This returns the strict JSON format required
        result = analyzer.process_frame(session_id, frame)

        # 1. Terminal Output (The raw Backend Data)
        # We only print if it's a valid dict (not the "Insufficient data" string)
        if isinstance(result, dict):
            # Print scores to terminal for verification
            scores = result["non_verbal_scores"]
            status = result["session_status"]
            print(f"Status: {status} | Score: {scores['final_non_verbal_score']} | Insights: {result['insights']}")

            # 2. Visual Overlay (For your verification)
            # Display results on the frame
            y0, dy = 30, 30
            cv2.putText(frame, f"Final Score: {scores['final_non_verbal_score']}", (10, y0), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            metrics = [
                f"Eye Contact: {scores['eye_contact']}",
                f"Expression: {scores['facial_expression']}",
                f"Posture: {scores['posture']}",
                f"Stability: {scores['stability']}"
            ]
            
            for i, line in enumerate(metrics):
                y = y0 + (i + 1) * dy
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            if status == "cancelled":
                cv2.putText(frame, "SESSION CANCELLED", (150, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.imshow('Non-Verbal Analysis Test', frame)
                cv2.waitKey(2000) # Show for 2 seconds before closing
                break
        else:
            cv2.putText(frame, "Waiting for Face...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the video
        cv2.imshow('Non-Verbal Analysis Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_test()
