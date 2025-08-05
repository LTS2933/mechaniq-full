import cv2
import numpy as np
import mediapipe as mp
import tempfile
import httpx
import os

mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180 / np.pi)
    return 360 - angle if angle > 180 else angle

async def analyze_video_from_url(url: str):
    print(f"🔗 Downloading video from URL: {url}")
    feedback = []
    biomarker_results = {}

    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            return [{"message": "Failed to download video"}], {}

        video_bytes = response.content

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [{"message": "Failed to read video file"}], {}

    pose = mp_pose.Pose(static_image_mode=False)
    frame_count = 0

    head_positions = []
    hip_angles, shoulder_angles = [], []
    wrist_positions = []
    attack_angles = []
    swing_speeds = []

    max_separation = 0
    peak_separation_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count > 30:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            try:
                l_shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                r_shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                l_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                r_hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                l_elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                l_wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x, lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                nose = [lm[mp_pose.PoseLandmark.NOSE.value].x, lm[mp_pose.PoseLandmark.NOSE.value].y]

                # Hip–shoulder rotation
                hip_angle = np.degrees(np.arctan2(r_hip[1] - l_hip[1], r_hip[0] - l_hip[0]))
                shoulder_angle = np.degrees(np.arctan2(r_shoulder[1] - l_shoulder[1], r_shoulder[0] - l_shoulder[0]))
                separation_angle = shoulder_angle - hip_angle

                hip_angles.append(hip_angle)
                shoulder_angles.append(shoulder_angle)

                if abs(separation_angle) > abs(max_separation):
                    max_separation = separation_angle
                    peak_separation_frame = frame_count

                # Wrist path and attack angle
                wrist_positions.append(l_wrist)
                if len(wrist_positions) > 1:
                    dx = wrist_positions[-1][0] - wrist_positions[-2][0]
                    dy = wrist_positions[-1][1] - wrist_positions[-2][1]
                    speed = np.sqrt(dx**2 + dy**2)
                    swing_speeds.append(speed)

                    # Approximate bat path vector
                    bat_vector = np.array(l_wrist) - np.array(l_elbow)
                    attack_angle = np.degrees(np.arctan2(-bat_vector[1], bat_vector[0]))
                    attack_angles.append(attack_angle)

                # Head stability
                head_positions.append(nose)

            except Exception as e:
                print(f"⚠️ Landmark error: {e}")

        frame_count += 1

    cap.release()
    os.remove(video_path)

    # Relative phase coordination
    hip_vel = np.gradient(hip_angles)
    shoulder_vel = np.gradient(shoulder_angles)
    relative_phase = np.arctan2(shoulder_vel, hip_vel)
    coordination_score = float(np.std(relative_phase))

    # Head movement std
    if len(head_positions) > 5:
        x_positions = [pos[0] for pos in head_positions]
        head_stability = float(np.std(x_positions))
    else:
        head_stability = 0.0

    # Feedback logic
    if abs(max_separation) < 15:
        feedback.append({
            "frame": peak_separation_frame,
            "issue": "Hip–shoulder separation below optimal range",
            "suggested_drill": "Practice coil drills to improve torque"
        })

    if len(attack_angles) > 0:
        avg_attack_angle = np.mean(attack_angles)
        if avg_attack_angle < 5 or avg_attack_angle > 25:
            feedback.append({
                "frame": "contact-phase",
                "issue": "Attack angle outside ideal range",
                "angle": round(avg_attack_angle, 2),
                "suggested_drill": "Use tee drills with upward focus to fix swing plane"
            })
    else:
        avg_attack_angle = None

    if coordination_score > 1.5:
        feedback.append({
            "frame": "multiple",
            "issue": "High relative-phase variance (poor upper-lower body timing)",
            "score": round(coordination_score, 3),
            "suggested_drill": "Use medicine ball throw drills to sync upper/lower body"
        })

    if head_stability > 0.02:
        feedback.append({
            "frame": "multiple",
            "issue": "Too much head movement during swing",
            "x_std_dev": round(head_stability, 4),
            "suggested_drill": "Practice dry swings in front of a mirror to keep head still"
        })

    biomarker_results = {
        "hip_shoulder_separation_max": round(max_separation, 2),
        "separation_peak_frame": peak_separation_frame,
        "coordination_score": round(coordination_score, 4),
        "head_stability": round(head_stability, 4),
        "attack_angle_avg": round(avg_attack_angle, 2) if avg_attack_angle else None,
        "swing_speed_peak": round(max(swing_speeds), 4) if swing_speeds else None
    }

    return feedback or [{"message": "No major issues detected."}], biomarker_results
