import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

class LungeAnalyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.total_reps = 0
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.lunge_stage = None  # 'standing', 'lunging'
        self.current_leg = None  # 'left', 'right'

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
        """Analyze the current frame for lunge form."""
        feedback = {}

        try:
            # Get coordinates for left and right legs
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

            # Calculate angles
            left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)

            # Determine which leg is forward based on the depth
            left_knee_forward = left_knee.z < right_knee.z
            right_knee_forward = right_knee.z < left_knee.z

            # Lunge detection logic
            if self.lunge_stage is None:
                self.lunge_stage = 'standing'
                self.current_leg = None

            if self.lunge_stage == 'standing':
                if left_knee_angle < 110 and left_knee_forward:
                    self.lunge_stage = 'lunging'
                    self.current_leg = 'left'
                elif right_knee_angle < 110 and right_knee_forward:
                    self.lunge_stage = 'lunging'
                    self.current_leg = 'right'

            elif self.lunge_stage == 'lunging':
                if left_knee_angle > 160 and right_knee_angle > 160:
                    self.lunge_stage = 'standing'
                    self.total_reps += 1

                    # Evaluate form and provide feedback
                    feedback_data = {
                        "count": self.total_reps,
                        "text": "",
                        "intent": 0
                    }

                    issues = []

                    # Check the front knee angle for proper bend
                    if self.current_leg == 'left':
                        front_knee_angle = left_knee_angle
                    else:
                        front_knee_angle = right_knee_angle

                    if not (80 <= front_knee_angle <= 110):
                        issues.append('Bend your front knee to about 90 degrees.')

                    # Check that the knee doesn't extend past the toes
                    front_ankle = left_ankle if self.current_leg == 'left' else right_ankle
                    front_knee = left_knee if self.current_leg == 'left' else right_knee
                    knee_over_toe = abs(front_knee.x - front_ankle.x)

                    if knee_over_toe > 0.1:
                        issues.append('Ensure your front knee does not go past your toes.')

                    # Provide feedback based on the evaluation
                    if issues:
                        self.incorrect_reps += 1
                        feedback_data["text"] = 'Incorrect lunge. ' + ' '.join(issues)
                        feedback_data["intent"] = 0
                    else:
                        self.correct_reps += 1
                        feedback_data["text"] = 'Good lunge!'
                        feedback_data["intent"] = 1

                    feedback["feedback"] = feedback_data

                    # Reset current leg
                    self.current_leg = None

            return feedback

        except Exception as e:
            feedback['error'] = f'Error in analyzing lunge: {e}'
            return feedback
