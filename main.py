import cv2
import mediapipe as mp
import numpy as np
import websockets
import asyncio
import json
import base64
from bicep_curl import BicepCurlAnalyzer
from lunge import LungeAnalyzer
from plank import PlankAnalyzer
from side_lateral_raise import SideLateralRaisesAnalyzer

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

# Utility function to calculate angles between three points (if needed)
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

# Process a single frame and generate feedback using BicepCurlAnalyzer
def process_frame(frame, analyzer):
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = analyzer.pose.process(rgb_frame)

        feedback = {}
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Analyze bicep curls
            feedback = analyzer.analyze(landmarks)

            # Prepare feedback data to send back to the client
            feedback_data = {
                "totalReps": analyzer.total_reps,
                "correctReps": analyzer.correct_reps,
                "incorrectReps": analyzer.incorrect_reps,
                "feedback":  feedback.get("feedback", ""),
                "error": feedback.get("error", "")
            }

            return feedback_data
        else:
            return {"error": "No pose detected"}
    except Exception as e:
        print(f"Error processing frame: {e}")
        return {"error": "Frame processing failed"}

# WebSocket server handler
async def server(websocket):  # Added 'path' parameter
    print("Client connected")

    # Create an instance of BicepCurlAnalyzer for this client
    analyzers = {
        'bicep_curl': BicepCurlAnalyzer(),
        'lunge': LungeAnalyzer(),
        'plank': PlankAnalyzer(),
        'lateral_raises': SideLateralRaisesAnalyzer()
    } 

    try:
        async for message in websocket:
            try:
                # Decode incoming frame
                data = json.loads(message)

                # Handle reset command
                if data.get("reset", False):
                    workout_name = data.get("workoutType")
                    if workout_name in analyzers:
                        analyzers[workout_name].reset()
                        print(f"Analyzer for {workout_name} reset.")
                        await websocket.send(json.dumps({"status": "Analyzer reset successful"}))
                    else:
                        await websocket.send(json.dumps({"error": "Invalid workout type for reset"}))
                    continue

                workout_name = data["workoutType"]
                analyzer=analyzers[workout_name]
                frame_data = base64.b64decode(data["frame"])
                np_frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

                # Process the frame and generate feedback
                feedback = process_frame(frame, analyzer)

                # Send feedback to the client
                await websocket.send(json.dumps(feedback))
            except Exception as e:
                print(f"Error handling message: {e}")
                await websocket.send(json.dumps({"error": "Message handling failed"}))
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        print("Client disconnected")

# Main function to start the WebSocket server
async def main():
    print("WebSocket server started on ws://localhost:8765")
    async with websockets.serve(server, "localhost", 8765):  # 'server' matches the handler
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
