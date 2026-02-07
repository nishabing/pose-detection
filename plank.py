import numpy as np
import mediapipe as mp
import time

mp_pose = mp.solutions.pose

class PlankAnalyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.feedback = ''
        self.total_reps = 0 
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.start_time = None
        self.correct_duration = 0

    def calculate_angle(self, a, b, c):
        """Calculate the angle between three points."""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def analyze(self, landmarks):
        """Analyze the current frame for plank form."""
        feedback = {}

        try:
            # Get coordinates for shoulders, hips, and ankles
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

            # Calculate angles
            left_shoulder_hip_ankle_angle = self.calculate_angle(left_shoulder, left_hip, left_ankle)
            right_shoulder_hip_ankle_angle = self.calculate_angle(right_shoulder, right_hip, right_ankle)

            # Average angles for both sides
            avg_shoulder_hip_ankle_angle = (left_shoulder_hip_ankle_angle + right_shoulder_hip_ankle_angle) / 2

            # Provide feedback based on the angles if the user is in a plank position
            if left_hip.visibility > 0.9 and right_hip.visibility > 0.9:  # Ensure user is detected
                feedback_data = {
                    "text": "",
                    "intent": 0
                }

                if 160 <= avg_shoulder_hip_ankle_angle <= 180:
                    feedback_data["text"] = "Good plank! Keep holding your body straight."
                    feedback_data["intent"] = 1
                    if self.start_time is None:
                        self.start_time = time.time()
                    else:
                        self.correct_duration = time.time() - self.start_time
                else:
                    feedback_data["text"] = "Incorrect plank. Ensure your body is in a straight line from shoulders to ankles."
                    feedback_data["intent"] = 0
                    self.start_time = None

                feedback["feedback"] = feedback_data
                feedback["correct_duration"] = self.correct_duration

            return feedback

        except Exception as e:
            feedback['error'] = f'Error in analyzing plank: {e}'
            return feedback