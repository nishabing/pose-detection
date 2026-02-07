import numpy as np
import mediapipe as mp
from utils import calculate_angle  # Ensure this utility function is implemented

mp_pose = mp.solutions.pose

class SideLateralRaisesAnalyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.reset()

    def reset(self):
        """Resets the analyzer state."""
        self.total_reps = 0
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.raise_stage = None  # Movement stage: "up" or "down"
        self.left_shoulder_angles = []
        self.right_shoulder_angles = []

        # Thresholds for movement validation
        self.SHOULDER_ANGLE_UP_THRESHOLD = 80    # Minimum angle for arms raised to shoulder height
        self.SHOULDER_ANGLE_REP_THRESHOLD = 60
        self.SHOULDER_ANGLE_DOWN_THRESHOLD = 30  # Maximum angle for arms in "down" position
        self.MAX_SHOULDER_ANGLE = 120            # Maximum acceptable angle to avoid overextension
        self.SHOULDER_POSITION_THRESHOLD = 0.02  # Movement stability threshold for shoulders

    def check_movement(self, positions, threshold):
        positions = np.array(positions)
        movement = np.max(positions, axis=0) - np.min(positions, axis=0)
        return np.all(movement < threshold)

    def analyze(self, landmarks):
        """Analyze the current frame for side lateral raise form."""
        feedback = {}

        # Get coordinates for left side
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

        # Get coordinates for right side
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # Calculate shoulder-to-elbow-to-wrist angles for left and right arms
        left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
        right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)

        # Initialize lists to store angles during a rep
        if self.raise_stage is None:
            self.left_shoulder_angles = []
            self.right_shoulder_angles = []

        # Lateral raise counter logic
        if left_shoulder_angle < self.SHOULDER_ANGLE_DOWN_THRESHOLD and \
           right_shoulder_angle < self.SHOULDER_ANGLE_DOWN_THRESHOLD :
            if (self.raise_stage=="up") :
                self.total_reps += 1
                self.left_shoulder_angles.append(left_shoulder_angle)
                self.right_shoulder_angles.append(right_shoulder_angle)

                # Evaluate the rep
                feedback_text = ""
                feedback_intent = 0
                issues = []

                overextension = False
                if np.max(self.left_shoulder_angles) > self.MAX_SHOULDER_ANGLE or np.max(self.right_shoulder_angles) > self.MAX_SHOULDER_ANGLE:
                    overextension = True
                    issues.append("Arms raised too high; avoid overextending above shoulder level.")

                # Check for insufficient lift
                insufficient_lift = False
                if np.max(self.left_shoulder_angles) < self.SHOULDER_ANGLE_UP_THRESHOLD or \
                np.max(self.right_shoulder_angles) < self.SHOULDER_ANGLE_UP_THRESHOLD:
                    insufficient_lift = True
                    issues.append("Arms not lifted high enough.")

                # Determine if the rep is correct or incorrect
                if not overextension and not insufficient_lift:
                    self.correct_reps += 1
                    feedback_text = "Good rep! Excellent form."
                    feedback_intent = 1
                else:
                    self.incorrect_reps += 1
                    feedback_text = "Incorrect rep due to: " + " ".join(issues)
                    feedback_intent = 0

                feedback["feedback"] = {
                    "count": self.total_reps,
                    "text": feedback_text,
                    "intent": feedback_intent,
                }

            self.raise_stage = "down"
            self.left_shoulder_angles= []
            self.right_shoulder_angles = []

        if left_shoulder_angle > self.SHOULDER_ANGLE_REP_THRESHOLD and \
           right_shoulder_angle > self.SHOULDER_ANGLE_REP_THRESHOLD:
            self.raise_stage = "up"
            # Collect positions during the movement
            self.left_shoulder_angles.append(left_shoulder_angle)
            self.right_shoulder_angles.append(right_shoulder_angle)

        return feedback
