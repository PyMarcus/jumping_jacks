import cv2
import mediapipe
import math
from typing import Any

"""
PS: this must use a lot of RAM memory
Because mediapipe uses tensor flow lib
"""


def view(content: Any) -> None:
    cv2.imshow('show', content)
    cv2.waitKey(40)


def create_video_instance(path: str) -> 'cv2.VideoCapture':
    return cv2.VideoCapture(path)


def solve_distance_from_points_hand(**kwargs) -> float:
    distance_from_hands = math.hypot(kwargs['hands_rx'] - kwargs['hands_lx'] -
                                     kwargs['hand_ry'] - kwargs['hand_ly'])
    return distance_from_hands


def solve_distance_from_points_feet(**kwargs) -> float:
    distance_from_feet = math.hypot(kwargs['foot_rx'] - kwargs['foot_lx'] -
                                    kwargs['foot_ry'] - kwargs['foot_ly'])
    return distance_from_feet


def monitor(path: str) -> None:
    video = create_video_instance(path=path)
    # detect pose from image
    pose = mediapipe.solutions.pose.Pose(min_tracking_confidence=.5, min_detection_confidence=.5)
    # draw lines and points
    draw = mediapipe.solutions.drawing_utils
    counter = 0

    while True:
        try:
            success, image = video.read()
            video_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # view(video_rgb)
            result = pose.process(video_rgb)
            points = result.pose_landmarks
            # image, points from pose landmarks and type of points
            draw.draw_landmarks(image, points, mediapipe.solutions.pose.POSE_CONNECTIONS)

            height, width, _ = image.shape
            if points:
                right_foot = int(points.landmark[mediapipe.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX].y * height)
                right_footx = int(points.landmark[mediapipe.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX].x * width)

                left_foot = int(points.landmark[mediapipe.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX].y * height)
                left_footx = int(points.landmark[mediapipe.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX].x * width)

                left_hand = int(points.landmark[mediapipe.solutions.pose.PoseLandmark.LEFT_INDEX].y * height)
                left_handx = int(points.landmark[mediapipe.solutions.pose.PoseLandmark.LEFT_INDEX].x * width)

                right_hand = int(points.landmark[mediapipe.solutions.pose.PoseLandmark.RIGHT_INDEX].y * height)
                right_handx = int(points.landmark[mediapipe.solutions.pose.PoseLandmark.RIGHT_INDEX].x * width)
                dist1 = solve_distance_from_points_hand(
                    hands_rx=right_handx,
                    hands_lx=left_handx,
                    hand_ry=right_hand,
                    hand_ly=left_hand
                )
                dist2 = solve_distance_from_points_feet(
                    foot_lx=left_footx,
                    foot_ry=right_foot,
                    foot_rx=right_footx,
                    foot_ly=left_foot
                )
                print(dist2, dist1)
                if (1467 <= dist2 <= 1517) and (370 <= dist1 <= 380):
                    counter += 1
                cv2.putText(image, str(counter), (10, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, 255)

                view(image)

                # ideal distance = 1517 328
        except Exception as e:
            ...


if __name__ == '__main__':
    monitor('video/jumpings.mp4')
