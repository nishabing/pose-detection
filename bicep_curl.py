import numpy as np
import mediapipe as mp
from utils import calculate_angle


mp_pose = mp.solutions.pose

class BicepCurlAnalyzer:
    def __init__(self):
        self.total_reps = 0
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.curl_stage = None  # 'down' or 'up'
        self.ELBOW_ANGLE_THRESHOLD = 30
        self.ELBOW_EXTENSION_THRESHOLD = 160
        self.ELBOW_POSITION_THRESHOLD = 0.03
        self.SHOULDER_POSITION_THRESHOLD = 0.02
        self.left_elbow_positions = []
        self.right_elbow_positions = []
        self.left_shoulder_positions = []
        self.right_shoulder_positions = []
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.9, min_tracking_confidence=0.9)


    def reset(self):
        """Reset all tracking attributes to start a new session."""
        self.total_reps = 0
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.curl_stage = None
        self.left_elbow_positions = []
        self.right_elbow_positions = []
        self.left_shoulder_positions = []
        self.right_shoulder_positions = []

    def check_movement(self, positions, threshold):
        """Check if movement is within acceptable thresholds."""
        positions = np.array(positions)
        movement = np.max(positions, axis=0) - np.min(positions, axis=0)
        movement_ok = np.all(movement < threshold)
        return movement_ok


    def is_elbow_close_to_body(self, shoulder_angle, threshold=20):
        return shoulder_angle < threshold
    
    def analyze(self, landmarks):
        """Analyze the current frame for bicep curl form."""
        feedback = {}

        # Get coordinates for left arm.
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        left_hip =landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

        # Get coordinates for right arm.
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        right_hip =landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]


        # Calculate angles.
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        left_shoulder_angle =calculate_angle(left_hip, left_shoulder, left_elbow)
        right_shoulder_angle =calculate_angle(right_hip, right_shoulder, right_elbow)

        # Initialize lists to store positions during a rep.
        if self.curl_stage is None:
            self.left_elbow_positions = []
            self.right_elbow_positions = []
            self.left_shoulder_positions = []
            self.right_shoulder_positions = []

        # Curl counter logic.
        if left_elbow_angle > self.ELBOW_EXTENSION_THRESHOLD and right_elbow_angle > self.ELBOW_EXTENSION_THRESHOLD:
            self.curl_stage = "down"
            self.left_elbow_positions = []
            self.right_elbow_positions = []
            self.left_shoulder_positions = []
            self.right_shoulder_positions = []
        if left_elbow_angle < self.ELBOW_ANGLE_THRESHOLD and right_elbow_angle < self.ELBOW_ANGLE_THRESHOLD and self.curl_stage == 'down':
            self.curl_stage = "up"
            self.total_reps += 1

            # Append the final positions.
            self.left_elbow_positions.append([left_elbow.x, left_elbow.y])
            self.right_elbow_positions.append([right_elbow.x, right_elbow.y])
            self.left_shoulder_positions.append([left_shoulder.x, left_shoulder.y])
            self.right_shoulder_positions.append([right_shoulder.x, right_shoulder.y])

            # Evaluate the rep.
            feedback_text = ''
            feedback_intent = 0
            issues = []

            # Analyze left arm movement.
            left_elbow_movement_ok = self.check_movement(self.left_elbow_positions, self.ELBOW_POSITION_THRESHOLD)
            left_shoulder_movement_ok = self.check_movement(self.left_shoulder_positions, self.SHOULDER_POSITION_THRESHOLD)

            # Analyze right arm movement.
            right_elbow_movement_ok = self.check_movement(self.right_elbow_positions, self.ELBOW_POSITION_THRESHOLD)
            right_shoulder_movement_ok = self.check_movement(self.right_shoulder_positions, self.SHOULDER_POSITION_THRESHOLD)

            left_elbow_close = self.is_elbow_close_to_body(left_shoulder_angle)
            right_elbow_close = self.is_elbow_close_to_body(right_shoulder_angle)

            # Determine if the rep is correct or incorrect.
            if left_elbow_movement_ok and left_shoulder_movement_ok and \
               right_elbow_movement_ok and right_shoulder_movement_ok:
                if left_elbow_close and right_elbow_close:
                    self.correct_reps += 1
                    feedback_text = 'Good rep! Excellent form on both arms.'
                    feedback_intent = 1
                else:
                    self.incorrect_reps += 1
                    feedback_text = 'Incorrect rep due to: Elbow not close to body'
                    feedback_intent = 0
                # self.correct_reps += 1
                # feedback_text = 'Good rep! Excellent form on both arms.'
                # feedback_intent = 1
            else:
                self.incorrect_reps += 1
                feedback_text = 'Incorrect rep due to:'
                if not left_elbow_movement_ok or not left_shoulder_movement_ok:
                    issues.append('Left arm improper movement.')
                if not right_elbow_movement_ok or not right_shoulder_movement_ok:
                    issues.append('Right arm improper movement.')
                if not left_elbow_close or not right_elbow_close:
                    issues.append('Elbow not close to body.')
                feedback_text += ' ' + ' '.join(issues)
                feedback_intent = 0

            feedback["feedback"] = {
                "count": self.total_reps,
                "text": feedback_text,
                "intent":feedback_intent,
            }

        else:
            # Collect positions during the movement.
            self.left_elbow_positions.append([left_elbow.x, left_elbow.y])
            self.right_elbow_positions.append([right_elbow.x, right_elbow.y])
            self.left_shoulder_positions.append([left_shoulder.x, left_shoulder.y])
            self.right_shoulder_positions.append([right_shoulder.x, right_shoulder.y])

        return feedback
