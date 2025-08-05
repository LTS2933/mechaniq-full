import cv2
import numpy as np
import mediapipe as mp
import tempfile
import httpx
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def draw_swing_phase_annotation(frame, phase_label, color=(0, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    position = (30, 50)
    cv2.putText(frame, phase_label, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return frame

def draw_frame_number(frame, frame_idx):
    text = f"Frame: {frame_idx}"
    position = (30, 100)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  # white
    thickness = 2
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return frame

def draw_separation(frame, separation_value):
    text = f"➤ Separation: {separation_value:.2f}°"
    position = (30, 150)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 255)
    thickness = 2
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return frame



def overlay_phase_label(frame_idx, swing_phases):
    labels = {
        "stride_start": "Stride Start",
        "foot_plant": "Foot Plant",
        "swing_start": "Swing Start",
        "contact": "Contact",
        "follow_through": "Follow Through"
    }
    for phase, idx in swing_phases.items():
        if idx == frame_idx:
            return labels.get(phase, "")
    return ""


async def generate_annotated_video(video_url: str, swing_phases: dict) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(video_url)
        if response.status_code != 200:
            raise Exception("Failed to download video")

        video_bytes = response.content

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        input_video_path = tmp.name

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise Exception("Failed to read video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "analyzed_output.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    pose = mp_pose.Pose(static_image_mode=False)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        phase_label = overlay_phase_label(frame_idx, swing_phases)
        if phase_label:
            frame = draw_swing_phase_annotation(frame, phase_label, color=(0, 255, 255))
            frame = draw_frame_number(frame, frame_idx)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    os.remove(input_video_path)
    return output_path


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180 / np.pi)
    return 360 - angle if angle > 180 else angle

def detect_handedness(landmarks_over_time, movement_threshold=0.005) -> str:
    """
    Determines handedness based on which ankle moves >= threshold (e.g., 0.05) first.
    Assumes the moving ankle is the lead leg → opposite of handedness.
    """
    left_ankle_x = []
    right_ankle_x = []

    for lm in landmarks_over_time:
        try:
            la = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x
            ra = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x
            left_ankle_x.append(la)
            right_ankle_x.append(ra)
        except:
            continue

    if not left_ankle_x or not right_ankle_x:
        return "unknown"

    initial_left = left_ankle_x[0]
    initial_right = right_ankle_x[0]

    left_stride_frame = None
    right_stride_frame = None

    for i in range(1, min(len(left_ankle_x), len(right_ankle_x))):
        if left_stride_frame is None and abs(left_ankle_x[i] - initial_left) > movement_threshold:
            left_stride_frame = i
        if right_stride_frame is None and abs(right_ankle_x[i] - initial_right) > movement_threshold:
            right_stride_frame = i
        if left_stride_frame is not None or right_stride_frame is not None:
            break  # Stop as soon as one moves enough

    print(f"👟 Left ankle moved at frame:  {left_stride_frame}")
    print(f"👟 Right ankle moved at frame: {right_stride_frame}")

    if left_stride_frame is not None and (right_stride_frame is None or left_stride_frame < right_stride_frame):
        return "right"
    elif right_stride_frame is not None:
        return "left"
    else:
        return "unknown"

def detect_swing_phases(landmarks_over_time, is_lefty=False):
    phases = {
        "stride_start": None,
        "foot_plant": None,
        "swing_start": None,
        "contact": None,
        "follow_through": None,
    }

    left_ankle_x = []
    right_ankle_x = []
    left_wrist_x = []
    left_elbow_x = []

    for i, lm in enumerate(landmarks_over_time):
        try:
            la = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            lw = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
            ra = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            le = lm[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            left_ankle_x.append(la.x)
            right_ankle_x.append(ra.x)
            left_wrist_x.append(lw.x)
            left_elbow_x.append(le.x)
        except:
            continue

    # STRIDE START
    lead_ankle_x = right_ankle_x if is_lefty else left_ankle_x
    initial_lead_ankle_x = lead_ankle_x[0]

    for i in range(1, len(lead_ankle_x)):
        movement = abs(lead_ankle_x[i] - initial_lead_ankle_x)
        print(f"Frame {i}: Lead ankle moved {movement:.5f}")
        if movement > 0.005:
            phases["stride_start"] = i
            break


    # FOOT PLANT
    if phases["stride_start"] is not None:
        knee_x = []
        for lm in landmarks_over_time:
            try:
                knee = lm[mp_pose.PoseLandmark.LEFT_KNEE.value if is_lefty else mp_pose.PoseLandmark.RIGHT_KNEE.value]
                knee_x.append(knee.x)
            except:
                knee_x.append(None)

        initial_knee_x = knee_x[phases["stride_start"]]
        
        for i in range(phases["stride_start"] + 1, len(knee_x)):
            if knee_x[i] is not None and abs(knee_x[i] - initial_knee_x) > 0.005:
                phases["foot_plant"] = i
                break

    if phases["foot_plant"] is not None:
        idx = phases["foot_plant"]
        try:
            lm = landmarks_over_time[idx]

            hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value if is_lefty else mp_pose.PoseLandmark.LEFT_HIP.value]
            opp_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value if is_lefty else mp_pose.PoseLandmark.RIGHT_HIP.value]

            shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value if is_lefty else mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            opp_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value if is_lefty else mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            hip_angle = np.degrees(np.arctan2(opp_hip.y - hip.y, opp_hip.x - hip.x))
            shoulder_angle = np.degrees(np.arctan2(opp_shoulder.y - shoulder.y, opp_shoulder.x - shoulder.x))
            separation = abs((shoulder_angle - hip_angle + 180) % 360 - 180)

            print(f"🧮 Foot Plant @ Frame {idx}")
            print(f"   ➤ Hip angle: {hip_angle:.2f}°")
            print(f"   ➤ Shoulder angle: {shoulder_angle:.2f}°")
            print(f"   ➤ Separation: {separation:.2f}°")

        except Exception as e:
            print(f"⚠️ Couldn't compute separation at foot plant: {e}")

    # SWING START
    for i in range(2, len(left_wrist_x)):
        dx = left_wrist_x[i] - left_wrist_x[i - 2]
        if dx > 0.03:
            phases["swing_start"] = i
            break

    # CONTACT
    for i in range((phases["swing_start"] or 2), len(left_wrist_x)):
        if left_wrist_x[i] - left_elbow_x[i] > 0.05:
            phases["contact"] = i
            break

    # FOLLOW THROUGH
    for i in range((phases["contact"] or len(left_wrist_x)), len(left_wrist_x)-2):
        if abs(left_wrist_x[i+2] - left_wrist_x[i]) < 0.01:
            phases["follow_through"] = i
            break

    return phases

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
    landmarks_over_time = []

    max_separation = 0
    peak_separation_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks_over_time.append(results.pose_landmarks.landmark)

        frame_count += 1

    cap.release()
    os.remove(video_path)

    # 🧠 Detect handedness and swing phases
    handedness = detect_handedness(landmarks_over_time)
    is_lefty = handedness == "left"
    print(f"🤚 Batter is detected as: {handedness}-handed")

    swing_phases = detect_swing_phases(landmarks_over_time, is_lefty=is_lefty)

    # 👉 Choose dynamic indices based on handedness
    shoulder_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER.value if is_lefty else mp_pose.PoseLandmark.LEFT_SHOULDER.value
    opp_shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER.value if is_lefty else mp_pose.PoseLandmark.RIGHT_SHOULDER.value

    hip_idx = mp_pose.PoseLandmark.RIGHT_HIP.value if is_lefty else mp_pose.PoseLandmark.LEFT_HIP.value
    opp_hip_idx = mp_pose.PoseLandmark.LEFT_HIP.value if is_lefty else mp_pose.PoseLandmark.RIGHT_HIP.value

    elbow_idx = mp_pose.PoseLandmark.RIGHT_ELBOW.value if is_lefty else mp_pose.PoseLandmark.LEFT_ELBOW.value
    wrist_idx = mp_pose.PoseLandmark.RIGHT_WRIST.value if is_lefty else mp_pose.PoseLandmark.LEFT_WRIST.value

    for i, lm in enumerate(landmarks_over_time):
        try:
            shoulder = [lm[shoulder_idx].x, lm[shoulder_idx].y]
            opp_shoulder = [lm[opp_shoulder_idx].x, lm[opp_shoulder_idx].y]
            hip = [lm[hip_idx].x, lm[hip_idx].y]
            opp_hip = [lm[opp_hip_idx].x, lm[opp_hip_idx].y]
            elbow = [lm[elbow_idx].x, lm[elbow_idx].y]
            wrist = [lm[wrist_idx].x, lm[wrist_idx].y]
            nose = [lm[mp_pose.PoseLandmark.NOSE.value].x, lm[mp_pose.PoseLandmark.NOSE.value].y]

            # 🖐️ Wrist movement
            wrist_positions.append(wrist)
            if len(wrist_positions) > 1:
                dx = wrist_positions[-1][0] - wrist_positions[-2][0]
                dy = wrist_positions[-1][1] - wrist_positions[-2][1]
                speed = np.sqrt(dx**2 + dy**2)
                swing_speeds.append(speed)

            # ⚾ Attack angle near contact
            if swing_phases["contact"] and i == swing_phases["contact"]:
                bat_vector = np.array(wrist) - np.array(elbow)
                attack_angle = np.degrees(np.arctan2(-bat_vector[1], bat_vector[0]))
                attack_angles.append(attack_angle)

            # 🧠 Head tracking
            if swing_phases["swing_start"] and i >= swing_phases["swing_start"]:
                head_positions.append(nose)

            # 💥 Separation at foot plant only
            if swing_phases["foot_plant"] and i == swing_phases["foot_plant"]:
                hip_angle = np.degrees(np.arctan2(opp_hip[1] - hip[1], opp_hip[0] - hip[0]))
                shoulder_angle = np.degrees(np.arctan2(opp_shoulder[1] - shoulder[1], opp_shoulder[0] - shoulder[0]))
                separation_angle = shoulder_angle - hip_angle
                hip_angles.append(hip_angle)
                shoulder_angles.append(shoulder_angle)
                max_separation = separation_angle
                peak_separation_frame = i

        except Exception as e:
            print(f"⚠️ Landmark error at frame {i}: {e}")

    # 🔗 Coordination score
    if len(hip_angles) >= 2 and len(shoulder_angles) >= 2:
        hip_vel = np.gradient(hip_angles)
        shoulder_vel = np.gradient(shoulder_angles)
        relative_phase = np.arctan2(shoulder_vel, hip_vel)
        coordination_score = float(np.std(relative_phase))
    else:
        coordination_score = 0.0


    # 🧠 Head movement stability
    head_stability = float(np.std([pos[0] for pos in head_positions])) if len(head_positions) > 5 else 0.0

    # ✅ Feedback
    if abs(max_separation) < 15:
        feedback.append({
            "frame": peak_separation_frame,
            "issue": "Hip–shoulder separation below optimal range",
            "suggested_drill": "Practice coil drills to improve torque"
        })

    if attack_angles:
        avg_attack_angle = np.mean(attack_angles)
        if avg_attack_angle < 5 or avg_attack_angle > 25:
            feedback.append({
                "frame": swing_phases["contact"],
                "issue": "Attack angle outside ideal range",
                "angle": round(avg_attack_angle, 2),
                "suggested_drill": "Use tee drills with upward focus to fix swing plane"
            })
    else:
        avg_attack_angle = None

    if coordination_score > 1.5:
        feedback.append({
            "frame": swing_phases["foot_plant"],
            "issue": "High relative-phase variance (poor upper-lower body timing)",
            "score": round(coordination_score, 3),
            "suggested_drill": "Use medicine ball throw drills to sync upper/lower body"
        })

    if head_stability > 0.02:
        feedback.append({
            "frame": "during swing",
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
        "swing_speed_peak": round(max(swing_speeds), 4) if swing_speeds else None,
        "swing_phases": swing_phases,
        "handedness": handedness
    }

    return feedback or [{"message": "No major issues detected."}], biomarker_results
