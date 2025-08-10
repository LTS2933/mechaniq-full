import cv2
import numpy as np
import mediapipe as mp
import tempfile
import httpx
import os
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import json
from typing import Dict, List, Tuple, Optional
import ffmpeg


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
PoseLandmark = mp_pose.PoseLandmark

class RobustBaseballSwingAnalyzer:
    """
    Enhanced baseball swing analyzer with more robust handedness detection
    and adaptive swing timing detection that handles various video lengths and zoom levels.
    """
    
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.body_proportions = None  # Will store normalized body measurements
    
    def calculate_adaptive_body_metrics(self, landmarks):
        """Calculate adaptive body metrics for normalization across different zoom levels."""
        try:
            # Primary measurements
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            
            # Calculate multiple reference measurements
            torso_length = np.sqrt((left_shoulder.x - left_hip.x) ** 2 + (left_shoulder.y - left_hip.y) ** 2)
            shoulder_width = np.sqrt((right_shoulder.x - left_shoulder.x) ** 2 + (right_shoulder.y - left_shoulder.y) ** 2)
            hip_width = np.sqrt((right_hip.x - left_hip.x) ** 2 + (right_hip.y - left_hip.y) ** 2)
            leg_length = np.sqrt((left_hip.x - left_ankle.x) ** 2 + (left_hip.y - left_ankle.y) ** 2)
            
            # Use composite measurement for better stability
            body_scale = np.mean([torso_length, shoulder_width * 2.5, hip_width * 3.0, leg_length * 0.8])
            
            # Hip center as reference point
            hip_center_x = (left_hip.x + right_hip.x) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2
            
            return {
                'scale': body_scale,
                'hip_center': (hip_center_x, hip_center_y),
                'torso_length': torso_length,
                'shoulder_width': shoulder_width,
                'measurements_valid': body_scale > 0.02
            }
        except:
            return None
    
    def normalize_landmark_position(self, landmark, reference_landmarks):
        """Enhanced normalization using adaptive body metrics."""
        try:
            metrics = self.calculate_adaptive_body_metrics(reference_landmarks)
            if not metrics or not metrics['measurements_valid']:
                return None
            
            hip_center_x, hip_center_y = metrics['hip_center']
            scale = metrics['scale']
            
            return {
                'x': (landmark.x - hip_center_x) / scale,
                'y': (landmark.y - hip_center_y) / scale,
                'raw_x': landmark.x,
                'raw_y': landmark.y,
                'scale': scale
            }
        except:
            return None


    def detect_handedness_fusion(self, landmarks_over_time):
        """
        Detect handedness based on which direction the face is turned.
        Uses relative positions of facial landmarks (e.g., eyes, ears, nose).
        If face points right → likely left-handed (facing camera left).
        If face points left → likely right-handed (facing camera right).
        """
        if len(landmarks_over_time) < 10:
            return {"handedness": "unknown", "confidence": 0.0, "debug": "Insufficient frames"}

        max_frames = min(40, len(landmarks_over_time))
        direction_scores = []

        for i in range(max_frames):
            lm = landmarks_over_time[i]

            try:
                nose = lm[mp_pose.PoseLandmark.NOSE.value]
                leye = lm[mp_pose.PoseLandmark.LEFT_EYE.value]
                reye = lm[mp_pose.PoseLandmark.RIGHT_EYE.value]
                lear = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
                rear = lm[mp_pose.PoseLandmark.RIGHT_EAR.value]

                if min(nose.visibility, leye.visibility, reye.visibility, lear.visibility, rear.visibility) < 0.5:
                    continue

                # Distances from nose to eyes/ears
                dist_nose_to_leye = nose.x - leye.x
                dist_nose_to_reye = reye.x - nose.x
                dist_nose_to_lear = nose.x - lear.x
                dist_nose_to_rear = rear.x - nose.x

                # Combine signals: negative means facing right (right-handed), positive means facing left (left-handed)
                direction_score = (
                    dist_nose_to_reye - dist_nose_to_leye +
                    dist_nose_to_rear - dist_nose_to_lear
                )
                direction_scores.append(direction_score)

            except Exception as e:
                continue

        if len(direction_scores) < 5:
            return {"handedness": "unknown", "confidence": 0.0, "debug": "Too few valid face frames"}

        avg_score = sum(direction_scores) / len(direction_scores)

        if abs(avg_score) < 0.01:
            return {"handedness": "unknown", "confidence": 0.0, "debug": "Face direction too ambiguous"}

        handedness = "left" if avg_score > 0 else "right"
        confidence = round(min(abs(avg_score) * 10, 1.0), 2)

        return {
            "handedness": handedness,
            "confidence": confidence,
            "debug": {
                "method": "face direction via relative eye/ear distances",
                "avg_direction_score": round(avg_score, 4),
                "frames_used": len(direction_scores)
            }
        }
    
    
    def detect_precise_swing_timing(self, landmarks_over_time, handedness):
        """
        Enhanced swing timing detection that adapts to different video lengths
        and zoom levels while maintaining accuracy.
        """
        video_length = len(landmarks_over_time)
        
        if video_length < 15:
            return {"stride_start": None, "foot_plant": None, "swing_start": None, "contact": None, "follow_through": None}
        
        is_lefty = handedness == "left"
        
        # Extract adaptive movement data based on video characteristics
        movement_data = self._extract_adaptive_movement_data(landmarks_over_time, is_lefty, video_length)
        
        if not movement_data:
            return {"stride_start": None, "foot_plant": None, "swing_start": None, "contact": None, "follow_through": None}
        
        phases = {}
        
        # Adaptive phase detection based on video length
        phases["stride_start"] = self._detect_stride_start_adaptive(movement_data, video_length)
        phases["foot_plant"] = self._detect_foot_plant_adaptive(movement_data, phases["stride_start"], video_length)
        phases["swing_start"] = self._detect_swing_start_adaptive(movement_data, phases["foot_plant"],video_length, handedness)
        phases["contact"] = self._detect_contact_adaptive(movement_data, phases["swing_start"], video_length, handedness)
        phases["follow_through"] = self._detect_follow_through_adaptive(movement_data, phases["contact"], video_length)
        
        return phases

    def _extract_adaptive_movement_data(self, landmarks_over_time, is_lefty, video_length):
        """Extract movement data with adaptive sampling based on video characteristics.
        Adds optional face center (median of available nose/eyes/ears) and keeps arrays aligned.
        """
        # Define body parts based on handedness
        if is_lefty:
            lead_side = "RIGHT"
            back_side = "LEFT"
        else:
            lead_side = "LEFT"
            back_side = "RIGHT"

        data = {
            'frames': [],
            'video_length': video_length,
            # Enhanced data tracking
            'lead_ankle_x': [], 'lead_ankle_y': [],
            'back_ankle_x': [], 'back_ankle_y': [],
            'lead_knee_x': [],  'lead_knee_y':  [],
            'back_knee_x': [],  'back_knee_y':  [],
            'lead_hip_x': [],   'lead_hip_y':   [],
            'back_hip_x': [],   'back_hip_y':   [],
            'lead_shoulder_x': [], 'lead_shoulder_y': [],
            'back_shoulder_x': [], 'back_shoulder_y': [],
            'lead_wrist_x': [], 'lead_wrist_y': [],
            'back_wrist_x': [], 'back_wrist_y': [],
            'lead_elbow_x': [], 'lead_elbow_y': [],
            'back_elbow_x': [], 'back_elbow_y': [],
            # Additional tracking for better analysis
            'body_scales': [],              # Track zoom level changes
            'hip_center_x': [], 'hip_center_y': [],
            # NEW: optional face center (kept aligned; may contain NaN)
            'face_center_x': [], 'face_center_y': [],
        }

        landmark_indices = {
            'lead_ankle':   getattr(mp_pose.PoseLandmark, f'{lead_side}_ANKLE').value,
            'back_ankle':   getattr(mp_pose.PoseLandmark, f'{back_side}_ANKLE').value,
            'lead_knee':    getattr(mp_pose.PoseLandmark, f'{lead_side}_KNEE').value,
            'back_knee':    getattr(mp_pose.PoseLandmark, f'{back_side}_KNEE').value,
            'lead_hip':     getattr(mp_pose.PoseLandmark, f'{lead_side}_HIP').value,
            'back_hip':     getattr(mp_pose.PoseLandmark, f'{back_side}_HIP').value,
            'lead_shoulder':getattr(mp_pose.PoseLandmark, f'{lead_side}_SHOULDER').value,
            'back_shoulder':getattr(mp_pose.PoseLandmark, f'{back_side}_SHOULDER').value,
            'lead_wrist':   getattr(mp_pose.PoseLandmark, f'{lead_side}_WRIST').value,
            'back_wrist':   getattr(mp_pose.PoseLandmark, f'{back_side}_WRIST').value,
            'lead_elbow':   getattr(mp_pose.PoseLandmark, f'{lead_side}_ELBOW').value,
            'back_elbow':   getattr(mp_pose.PoseLandmark, f'{back_side}_ELBOW').value,
        }

        # Face landmarks (optional; we won't reject the frame if these are missing)
        face_indices = {
            'nose':  mp_pose.PoseLandmark.NOSE.value,
            'l_eye': mp_pose.PoseLandmark.LEFT_EYE.value,
            'r_eye': mp_pose.PoseLandmark.RIGHT_EYE.value,
            'l_ear': mp_pose.PoseLandmark.LEFT_EAR.value,
            'r_ear': mp_pose.PoseLandmark.RIGHT_EAR.value,
        }

        for frame_idx, landmarks in enumerate(landmarks_over_time):
            try:
                frame_data = {}
                all_present = True

                # 1) Body metrics for normalization
                body_metrics = self.calculate_adaptive_body_metrics(landmarks)
                if not body_metrics or not body_metrics.get('measurements_valid', False):
                    continue  # skip frame if we can't normalize reliably

                # 2) REQUIRED body joints (normalized); if any missing, skip the frame
                for part_name, landmark_idx in landmark_indices.items():
                    norm = self.normalize_landmark_position(landmarks[landmark_idx], landmarks)
                    if norm:
                        frame_data[f'{part_name}_x'] = norm['x']
                        frame_data[f'{part_name}_y'] = norm['y']
                    else:
                        all_present = False
                        break

                if not all_present:
                    continue

                # 3) OPTIONAL face center (median of available nose/eyes/ears); keep frame even if missing
                face_x_vals, face_y_vals = [], []
                for key in ('nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear'):
                    li = face_indices[key]
                    norm = self.normalize_landmark_position(landmarks[li], landmarks)
                    if norm is not None and np.isfinite(norm.get('x', np.nan)) and np.isfinite(norm.get('y', np.nan)):
                        face_x_vals.append(norm['x'])
                        face_y_vals.append(norm['y'])

                if face_x_vals:
                    face_cx = float(np.median(face_x_vals))
                    face_cy = float(np.median(face_y_vals))
                else:
                    face_cx = float('nan')
                    face_cy = float('nan')

                # 4) ACCEPT the frame: append everything in lockstep so arrays stay aligned
                data['frames'].append(frame_idx)

                data['body_scales'].append(body_metrics['scale'])
                data['hip_center_x'].append(body_metrics['hip_center'][0])
                data['hip_center_y'].append(body_metrics['hip_center'][1])

                for key, value in frame_data.items():
                    data[key].append(value)

                data['face_center_x'].append(face_cx)
                data['face_center_y'].append(face_cy)

            except Exception:
                # Be robust to occasional frames with odd data
                continue

        # Require a minimum number of accepted frames
        return data if len(data['frames']) > max(8, video_length // 6) else None

    
    def _detect_stride_start_adaptive(self, data, video_length):
        """Robust stride start detection using confidence scoring based on multi-joint signals."""
        if len(data['lead_ankle_x']) < 8:
            return None

        fps = len(data['frames']) / video_length if video_length > 0 else 30

        # Baseline period (setup)
        baseline_length = max(6, int(fps * 0.2))  # first 0.2s
        baseline_ankle_x = np.median(data['lead_ankle_x'][:baseline_length])
        baseline_ankle_y = np.median(data['lead_ankle_y'][:baseline_length])
        baseline_knee_x = np.median(data['lead_knee_x'][:baseline_length])

        scale_variation = np.std(data['body_scales'][:baseline_length]) if len(data['body_scales']) > baseline_length else 0.02
        ankle_range_x = np.std(data['lead_ankle_x'][:baseline_length])
        ankle_range_y = np.std(data['lead_ankle_y'][:baseline_length])
        threshold_x = max(0.08, ankle_range_x * 3, scale_variation * 2)
        threshold_y = max(0.06, ankle_range_y * 2.5, scale_variation * 1.5)

        # Smoothing
        sigma = max(1.0, video_length / 80)
        ankle_x = gaussian_filter1d(data['lead_ankle_x'], sigma=sigma)
        ankle_y = gaussian_filter1d(data['lead_ankle_y'], sigma=sigma)
        knee_x = gaussian_filter1d(data['lead_knee_x'], sigma=sigma)

        best_frame = None
        best_score = 0

        for i in range(baseline_length + 2, len(ankle_x) - 3):
            score = 0.0

            dx = abs(ankle_x[i] - baseline_ankle_x)
            dy = abs(ankle_y[i] - baseline_ankle_y)
            dk = abs(knee_x[i] - baseline_knee_x)

            # 1. Ankle starts moving forward/down
            if dx > threshold_x * 0.6:
                score += 0.3
            if dy > threshold_y * 0.5:
                score += 0.3

            # 2. Knee movement helps confirm stride
            if dk > 0.05:
                score += 0.2

            # 3. Acceleration of ankle
            ax = (ankle_x[i] - ankle_x[i-2]) / 2
            ay = (ankle_y[i] - ankle_y[i-2]) / 2
            accel = np.sqrt(ax**2 + ay**2)
            if accel > 0.01:
                score += 0.2

            # 4. Sustained movement (forward motion continues)
            sustained = abs(ankle_x[i+2] - ankle_x[i]) > 0.03
            if sustained:
                score += 0.2

            # 5. Timing bonus — if it's early in the swing (~first 1/3)
            if i < len(ankle_x) // 3:
                score += 0.1

            # Keep best-scoring frame
            if score >= 0.6:
                return data['frames'][i]

            if score > best_score:
                best_score = score
                best_frame = data['frames'][i]

        # Fallback to best guess
        if best_score > 0.4:
            return best_frame

        return None
    



    def _detect_foot_plant_adaptive(self, data, stride_start, video_length):
        """
        Detects foot plant using dual-ankle comparison with intelligent fallback to best candidate.
        """
        if stride_start is None or len(data['lead_ankle_y']) < 10:
            print("🚫 Not enough ankle data or stride_start is None.")
            return None

        if 'back_ankle_y' not in data or len(data['back_ankle_y']) != len(data['lead_ankle_y']):
            print("⚠️ No trail ankle data available, falling back to single-ankle method.")
            return None

        try:
            stride_index = next(i for i, f in enumerate(data['frames']) if f >= stride_start)
        except StopIteration:
            print("⚠️ Stride start not found in frames, using index 0.")
            stride_index = 0

        fps = len(data['frames']) / video_length if video_length > 0 else 30
        fps = max(fps, 10)
        smoothing_sigma = max(0.8, fps / 60)

        smoothed_lead_y = gaussian_filter1d(data['lead_ankle_y'], sigma=smoothing_sigma)
        smoothed_back_y = gaussian_filter1d(data['back_ankle_y'], sigma=smoothing_sigma)
        ankle_separation = smoothed_back_y - smoothed_lead_y

        search_start = stride_index
        search_duration = int(4.5 * fps)
        search_end = min(search_start + search_duration, len(smoothed_lead_y) - 10)

        min_stability_frames = max(2, int(0.08 * fps))
        max_stability_frames = max(10, int(0.5 * fps))
        stability_threshold = 0.0008

        baseline_abs_separation = np.median(np.abs(ankle_separation[stride_index:stride_index + min(20, len(ankle_separation)//4)]))
        convergence_threshold = min(0.02, baseline_abs_separation * 0.3)

        #print(f"🦶 Convergence threshold: {convergence_threshold:.4f}")
        #print(f"🕒 Stability: {min_stability_frames}-{max_stability_frames} frames | σ: {smoothing_sigma:.2f} | FPS: {fps:.2f}")

        best_score = -1
        best_frame = None

        for i in range(search_start + 1, search_end - max_stability_frames):
            prev_lead = smoothed_lead_y[i - 1]
            curr_lead = smoothed_lead_y[i]
            next_lead = smoothed_lead_y[i + 1]
            is_local_min = prev_lead > curr_lead < next_lead

            curr_sep = ankle_separation[i]
            curr_abs_sep = abs(curr_sep)
            feet_converged = curr_abs_sep < convergence_threshold

            # Check nearby for convergence
            feet_converged_nearby = (
                feet_converged or
                (i > 0 and abs(ankle_separation[i - 1]) < convergence_threshold) or
                (i + 1 < len(ankle_separation) and abs(ankle_separation[i + 1]) < convergence_threshold)
            )

            score = 0
            if is_local_min:
                score += 1
            if feet_converged_nearby:
                score += 1

            # Try to find a stability window
            for window_size in range(min_stability_frames, max_stability_frames + 1):
                if i + window_size >= len(smoothed_lead_y):
                    break

                lead_var = np.var(smoothed_lead_y[i:i + window_size])
                sep_window = np.abs(ankle_separation[i:i + window_size])
                sep_var = np.var(sep_window)
                sep_mean = np.mean(sep_window)

                lead_stable = lead_var < stability_threshold
                conv_stable = sep_var < stability_threshold * 1.5
                conv_maintained = sep_mean < convergence_threshold * 1.2

                if lead_stable:
                    score += 0.5
                if conv_stable and conv_maintained:
                    score += 0.5

                if score >= 2.0:
                    print(f"🎯 FOOT PLANT ~CONFIRMED at frame {data['frames'][i]} (score {score})")
                    return data['frames'][i]
                break  # Exit after 1 window attempt to keep search fast

            if score > best_score:
                best_score = score
                best_frame = data['frames'][i]

            print(f"🧪 Frame {data['frames'][i]}: local_min={is_local_min}, converged_nearby={feet_converged_nearby}, score={score}")

        print("⚠️ No perfect foot plant frame found. Best guess:")

        if best_score >= 1.5:
            print(f"✅ Using best candidate frame {best_frame} with score {best_score}")
            return best_frame

        print("🚫 No sufficiently confident foot plant frame.")
        return None

    def _detect_swing_start_adaptive(
        self,
        data,
        foot_plant_frame: int | None,
        video_length: int,
        handedness: str,
        *,
        fps: float | None = None,           # only used to cap to ~3s
        smooth_sigma: float = 0.6,
        min_dx_units: float = 0.0015,
        min_center_step_units: float = 0.001
    ):
        if foot_plant_frame is None:
            print("❌ No foot plant frame provided. Cannot determine swing start.")
            return None

        def _smooth1d(x, sigma):
            x = np.asarray(x, float)
            if sigma and sigma > 0:
                n = len(x); idx = np.arange(n); m = np.isfinite(x)
                if n and (not m.all()) and m.any():
                    x = x.copy(); x[~m] = np.interp(idx[~m], idx[m], x[m])
                from scipy.ndimage import gaussian_filter1d
                x = gaussian_filter1d(x, sigma=sigma)
            return x

        frames = np.asarray(data["frames"])
        plant_i = int(np.searchsorted(frames, foot_plant_frame, side="left"))
        if plant_i >= len(frames) - 2:
            print("❌ Not enough frames after foot plant.")
            return None

        # --- cap search to ~3 seconds after FP (or 90 frames if fps unknown) ---
        max_after = int(3 * fps) if (fps and fps > 0) else 90
        search_end_i = min(len(frames) - 1, plant_i + max_after)
        print(f"🔎 Swing-start scan window: idx[{plant_i+1}..{search_end_i}]")

        # --- mid-hands and mid-shoulders (smoothed) ---
        Lx = _smooth1d(data["lead_wrist_x"], smooth_sigma)
        Bx = _smooth1d(data["back_wrist_x"],  smooth_sigma)
        Hx = 0.5*(Lx + Bx)

        Lsx = _smooth1d(data["lead_shoulder_x"], 0.8)
        Bsx = _smooth1d(data["back_shoulder_x"], 0.8)
        Sx  = 0.5*(Lsx + Bsx)

        # OPTIONAL: blend in hip center to make the “center line” even stabler
        # Hx_c = np.asarray(data.get("hip_center_x", [np.nan]*len(Sx)), float)
        # if np.isfinite(Hx_c).sum() > len(Sx)*0.7:
        #     center_series = 0.5*Sx + 0.5*_smooth1d(Hx_c, 1.0)
        # else:
        #     center_series = Sx

        center_series = Sx  # using shoulders only for now

        # --- FIXED center at foot plant ---
        center_fp = float(center_series[plant_i])

        # direction gate (same as before)
        sign = +1.0 if handedness == "right" else (-1.0 if handedness == "left" else None)

        # --- first qualifying inward X step after FP (toward fixed FP center) ---
        start_i = plant_i + 1
        for i in range(start_i, search_end_i + 1):
            dx = Hx[i] - Hx[i-1]

            # distance to fixed FP center, not the moving torso center
            cx_prev = abs(Hx[i-1] - center_fp)
            cx_curr = abs(Hx[i]   - center_fp)
            toward_center = (cx_curr <= cx_prev - abs(min_center_step_units))

            dir_ok = (sign is None) or (sign * dx >= abs(min_dx_units))

            #print(
              #  f"🧪 idx {i}: Hx={Hx[i]:.4f}, Cfp={center_fp:.4f}, dx={dx:.5f}, "
              #  f"|Hx-Cfp|: {cx_prev:.4f}→{cx_curr:.4f} ({'↓' if toward_center else '↔/↑'}), "
              #  f"dir_ok={dir_ok}"
            # )

            if toward_center and dir_ok:
                frame = int(frames[i])
                print(f"✅ Swing start (first inward X step to FP center) at frame {frame} (idx {i})")
                return frame

        print("🔻 No inward step toward FP center found within the capped window.")
        return None
    
    def _detect_contact_adaptive(self, data, swing_start, video_length, handedness: str):
        """Contact ≈ frame within ~N seconds after swing_start where the lead arm is close to fully straight.
        Priority: earliest frame where hands are ahead of face (given handedness) AND extension ≥ near_thresh.
        Fallbacks: closest-to-full among near-full frames; else max extension in window.
        """
        if swing_start is None:
            print("❌ _detect_contact_adaptive: swing_start is None")
            return None

        # --- locate swing_start index in data["frames"] ---
        start_idx = 0
        for i, frame in enumerate(data['frames']):
            if frame >= swing_start:
                start_idx = i
                break

        # --- fixed window after swing start (assume 30 fps) ---
        WINDOW_SECONDS = 4.0
        fps_est = 30.0
        window_frames = int(round(fps_est * WINDOW_SECONDS))
        end_idx = min(start_idx + window_frames, len(data['lead_wrist_x']) - 1)
        print(
            f"🔎 contact: swing_start={swing_start}, start_idx={start_idx}, "
            f"fps_assumed={fps_est:.1f}, window_seconds={WINDOW_SECONDS}, "
            f"window_frames={window_frames}, end_idx={end_idx}"
        )

        if end_idx - start_idx < 6:
            print("🚫 contact: too few frames in window after swing_start")
            return None

        # --- smoothing helper ---
        def _smooth1d(x, sigma):
            x = np.asarray(x, float)
            if sigma and sigma > 0:
                n = len(x); idx = np.arange(n); m = np.isfinite(x)
                if n and (not m.all()) and m.any():
                    x = x.copy(); x[~m] = np.interp(idx[~m], idx[m], x[m])
                x = gaussian_filter1d(x, sigma=sigma)
            return x

        # ---------- series (smoothed) ----------
        lsx = _smooth1d(data['lead_shoulder_x'], 0.8)
        lsy = _smooth1d(data['lead_shoulder_y'], 0.8)
        lex = _smooth1d(data['lead_elbow_x'],   0.8)
        ley = _smooth1d(data['lead_elbow_y'],   0.8)
        lwx = _smooth1d(data['lead_wrist_x'],   0.8)
        lwy = _smooth1d(data['lead_wrist_y'],   0.8)

        # lead arm extension & arm length estimate
        upper_len = np.sqrt((lex - lsx)**2 + (ley - lsy)**2)
        fore_len  = np.sqrt((lwx - lex)**2 + (lwy - ley)**2)
        arm_len_est = np.median((upper_len[start_idx:end_idx] + fore_len[start_idx:end_idx]))
        ext = np.sqrt((lwx - lsx)**2 + (lwy - lsy)**2)  # shoulder→wrist
        ext_s = _smooth1d(ext, 0.8)

        # baseline to avoid micro false-positives near swing start
        lo, hi = start_idx, end_idx
        pre_span = max(3, min(6, end_idx - start_idx))
        baseline_ext = float(np.median(ext_s[start_idx:start_idx + pre_span]))
        ext_seg = ext_s[lo:hi+1]
        if ext_seg.size == 0:
            print("🚫 contact: empty ext_seg slice")
            return None

        # thresholds for "near full" candidate
        NEAR_RATIO = 0.92
        MIN_GAIN   = 0.03
        if not np.isfinite(arm_len_est) or arm_len_est <= 0:
            near_thresh = baseline_ext + MIN_GAIN
        else:
            near_thresh = max(NEAR_RATIO * arm_len_est, baseline_ext + MIN_GAIN)

        # ---------- FACE CHECK: hands ahead of face given handedness ----------
        hx = 0.5 * (np.asarray(data['lead_wrist_x'], float) + np.asarray(data['back_wrist_x'], float))
        hx_s = _smooth1d(hx, 0.6)
        face_x = np.asarray(data.get('face_center_x', []), float)
        have_face = (len(face_x) == len(hx_s))
        face_x_s = _smooth1d(face_x, 0.8) if have_face else face_x

        sign = +1.0 if handedness == 'right' else (-1.0 if handedness == 'left' else None)
        margin = 0.01  # tiny buffer in normalized units

        if not have_face:
            print("⚠️ face-gate: face_center_x not available or misaligned; skipping face check")
        elif sign is None:
            print("⚠️ face-gate: handedness unknown; skipping face check")
        else:
            fin = np.isfinite(face_x_s[lo:hi+1]).sum()
            print(
                f"🧠 face-gate: have_face=True, handedness={handedness}, "
                f"finite_face_in_window={fin}/{(hi-lo+1)}"
            )

        def hands_ahead(idx: int) -> bool:
            if not (have_face and sign is not None):
                return False
            if not (np.isfinite(hx_s[idx]) and np.isfinite(face_x_s[idx])):
                return False
            diff = sign * (hx_s[idx] - face_x_s[idx])
            return diff >= margin

        # --- DEBUG: Per-frame extension and face-check info ---
        print("\n🛠 DEBUG: Frame-by-frame contact search window")
        print(" idx | frame | ext_len  | ratio_to_full | near_thresh? | hands_ahead?")
        print("-----|-------|----------|---------------|--------------|--------------")
        for idx in range(lo, hi + 1):
            frame_num = int(data['frames'][idx])
            ext_len = ext_s[idx]
            ratio = (ext_len / arm_len_est) if (arm_len_est and np.isfinite(arm_len_est) and arm_len_est != 0) else float('nan')
            near_full = bool(np.isfinite(ext_len) and ext_len >= near_thresh)
            ahead = hands_ahead(idx)
            print(f"{idx:4d} | {frame_num:5d} | {ext_len:8.4f} | "
                f"{ratio:>13.3f} | {str(near_full):>12} | {str(ahead):>12}")

        print(
            f"📏 arm_len_est={arm_len_est:.4f}, baseline_ext={baseline_ext:.4f}, "
            f"near_thresh={near_thresh:.4f} (ratio={NEAR_RATIO}, min_gain={MIN_GAIN})"
        )
        print(
            f"   ext in window: min={np.nanmin(ext_seg):.4f}, max={np.nanmax(ext_seg):.4f}, "
            f"@frame={int(data['frames'][lo + int(np.nanargmax(ext_seg))])}"
        )

        # =========================
        # SELECTION LOGIC (updated)
        # =========================

        # STEP 1: Earliest frame with hands ahead AND near-full extension
        earliest_ahead_idx = None
        for idx in range(lo, hi + 1):
            if np.isfinite(ext_s[idx]) and ext_s[idx] >= near_thresh and hands_ahead(idx):
                earliest_ahead_idx = idx
                break

        if earliest_ahead_idx is not None:
            k = earliest_ahead_idx
            print(
                f"🎯 contact: earliest ahead-of-face & near-full at idx={k} "
                f"(frame={int(data['frames'][k])}), ext={ext_s[k]:.4f}"
            )
            return int(data['frames'][k])

        # STEP 2: Fallback — closest-to-full among near-full frames
        cand_mask = np.isfinite(ext_seg) & (ext_seg >= near_thresh)
        if np.any(cand_mask):
            cand_idxs_rel = np.where(cand_mask)[0]
            cand_idxs_abs = [lo + r for r in cand_idxs_rel]
            k = min(cand_idxs_abs, key=lambda i: abs(ext_s[i] - arm_len_est))
            print(
                f"ℹ️ no ahead-of-face near-full; using closest-to-full idx={k} "
                f"(frame={int(data['frames'][k])}), ext={ext_s[k]:.4f}"
            )
        else:
            # STEP 3: Fallback — max extension in window
            rel_k = int(np.nanargmax(ext_seg))
            k = lo + rel_k
            print(
                f"ℹ️ no near-full candidates; using max extension idx={k} "
                f"(frame={int(data['frames'][k])}), ext={ext_s[k]:.4f}"
            )

        # Optional nudge for fallback pick: look ahead a few frames to see if hands become ahead
        if have_face and sign is not None:
            nudged = None
            for t in range(k + 1, min(k + 6, end_idx + 1)):
                ahead = hands_ahead(t)
                close_enough = bool(np.isfinite(ext_s[t]) and ext_s[t] >= near_thresh)
                hv, fv = hx_s[t], face_x_s[t]
                diff_t = sign * (hv - fv) if (np.isfinite(hv) and np.isfinite(fv)) else np.nan
                print(
                    f"   ↪️ try t={t} (frame={int(data['frames'][t])}): "
                    f"ext={ext_s[t]:.4f} ({(ext_s[t]/arm_len_est if arm_len_est else float('nan')):.3f}×arm), "
                    f"close_enough={close_enough}, "
                    f"hx={hv:.4f} fx={fv:.4f} sign*Δ={diff_t if np.isfinite(diff_t) else float('nan'):.4f} "
                    f"ahead={ahead}"
                )
                if close_enough and ahead:
                    nudged = t
                    break

            if nudged is not None:
                print(
                    f"🎯 contact: nudged to idx={nudged} (frame={int(data['frames'][nudged])}) — returning"
                )
                return int(data['frames'][nudged])

        # Final return (fallback candidate)
        print(
            f"✅ contact: returning idx={k} (frame={int(data['frames'][k])}); "
            f"no earlier ahead-of-face near-full found"
        )
        return int(data['frames'][k])


    
    def _find_maximum_hand_separation_adaptive(self, data, start_idx, search_window):
        """Enhanced maximum hand separation detection."""
        max_separation = 0
        max_frame = None
        
        # Use smoothed data for more stable detection
        lead_wrist_x_smooth = gaussian_filter1d(data['lead_wrist_x'][start_idx:start_idx+search_window], sigma=0.8)
        lead_wrist_y_smooth = gaussian_filter1d(data['lead_wrist_y'][start_idx:start_idx+search_window], sigma=0.8)
        back_wrist_x_smooth = gaussian_filter1d(data['back_wrist_x'][start_idx:start_idx+search_window], sigma=0.8)
        back_wrist_y_smooth = gaussian_filter1d(data['back_wrist_y'][start_idx:start_idx+search_window], sigma=0.8)
        
        for i in range(len(lead_wrist_x_smooth)):
            separation = np.sqrt(
                (lead_wrist_x_smooth[i] - back_wrist_x_smooth[i])**2 +
                (lead_wrist_y_smooth[i] - back_wrist_y_smooth[i])**2
            )
            
            if separation > max_separation:
                max_separation = separation
                max_frame = data['frames'][start_idx + i]
        
        return max_frame
    
    def _find_maximum_bat_speed_adaptive(self, data, start_idx, search_window):
        """Enhanced maximum bat speed detection."""
        max_speed = 0
        max_frame = None
        
        # Smooth hand positions
        smoothed_x = gaussian_filter1d(data['lead_wrist_x'][start_idx:start_idx+search_window], sigma=0.6)
        smoothed_y = gaussian_filter1d(data['lead_wrist_y'][start_idx:start_idx+search_window], sigma=0.6)
        
        for i in range(3, len(smoothed_x) - 3):
            dx = smoothed_x[i+3] - smoothed_x[i-3]
            dy = smoothed_y[i+3] - smoothed_y[i-3]
            speed = np.sqrt(dx**2 + dy**2) / 6  # 6-frame window
            
            if speed > max_speed:
                max_speed = speed
                max_frame = data['frames'][start_idx + i]
        
        return max_frame
    
    def _find_lead_arm_extension_max_adaptive(self, data, start_idx, search_window):
        """Enhanced lead arm extension detection."""
        max_extension = 0
        max_frame = None
        
        # Smooth positions for stable measurement
        lead_shoulder_x = gaussian_filter1d(data['lead_shoulder_x'][start_idx:start_idx+search_window], sigma=0.8)
        lead_shoulder_y = gaussian_filter1d(data['lead_shoulder_y'][start_idx:start_idx+search_window], sigma=0.8)
        lead_wrist_x = gaussian_filter1d(data['lead_wrist_x'][start_idx:start_idx+search_window], sigma=0.8)
        lead_wrist_y = gaussian_filter1d(data['lead_wrist_y'][start_idx:start_idx+search_window], sigma=0.8)
        
        for i in range(len(lead_shoulder_x)):
            extension = np.sqrt(
                (lead_wrist_x[i] - lead_shoulder_x[i])**2 +
                (lead_wrist_y[i] - lead_shoulder_y[i])**2
            )
            
            if extension > max_extension:
                max_extension = extension
                max_frame = data['frames'][start_idx + i]
        
        return max_frame
    
    def _find_kinetic_sequence_peak_adaptive(self, data, start_idx, search_window):
        """Enhanced kinetic sequence peak detection."""
        max_sequence_value = 0
        max_frame = None
        
        # Calculate combined hip and shoulder velocities
        hip_angles = []
        shoulder_angles = []
        
        for i in range(start_idx, start_idx + search_window):
            if i >= len(data['lead_hip_x']):
                break
                
            hip_angle = np.arctan2(
                data['back_hip_y'][i] - data['lead_hip_y'][i],
                data['back_hip_x'][i] - data['lead_hip_x'][i]
            )
            shoulder_angle = np.arctan2(
                data['back_shoulder_y'][i] - data['lead_shoulder_y'][i],
                data['back_shoulder_x'][i] - data['lead_shoulder_x'][i]
            )
            
            hip_angles.append(hip_angle)
            shoulder_angles.append(shoulder_angle)
        
        if len(hip_angles) < 6:
            return None
        
        # Smooth angles
        hip_smooth = gaussian_filter1d(hip_angles, sigma=1.0)
        shoulder_smooth = gaussian_filter1d(shoulder_angles, sigma=1.0)
        
        for i in range(3, len(hip_smooth) - 3):
            hip_velocity = abs((hip_smooth[i+2] - hip_smooth[i-2]) / 4)
            shoulder_velocity = abs((shoulder_smooth[i+2] - shoulder_smooth[i-2]) / 4)
            
            # Combined sequence value (both rotating fast)
            sequence_value = hip_velocity * 0.6 + shoulder_velocity * 0.4
            
            if sequence_value > max_sequence_value:
                max_sequence_value = sequence_value
                max_frame = data['frames'][start_idx + i]
        
        return max_frame
    
    def _find_weight_transfer_completion_adaptive(self, data, start_idx, search_window):
        """Detect completion of weight transfer to front leg."""
        # Look for stabilization of lead leg position
        lead_knee_x = data['lead_knee_x'][start_idx:start_idx+search_window]
        lead_ankle_x = data['lead_ankle_x'][start_idx:start_idx+search_window]
        
        if len(lead_knee_x) < 6:
            return None
        
        # Smooth data
        knee_smooth = gaussian_filter1d(lead_knee_x, sigma=1.0)
        ankle_smooth = gaussian_filter1d(lead_ankle_x, sigma=1.0)
        
        min_movement = float('inf')
        min_frame = None
        
        for i in range(3, len(knee_smooth) - 3):
            # Calculate movement in lead leg
            knee_movement = abs((knee_smooth[i+2] - knee_smooth[i-2]) / 4)
            ankle_movement = abs((ankle_smooth[i+2] - ankle_smooth[i-2]) / 4)
            
            total_movement = knee_movement + ankle_movement
            
            # Weight transfer completion = minimal lead leg movement
            if total_movement < min_movement:
                min_movement = total_movement
                min_frame = data['frames'][start_idx + i]
        
        # Only return if movement is sufficiently low
        return min_frame if min_movement < 0.03 else None
    
    def _detect_follow_through_adaptive(self, data, contact, video_length):
        """Enhanced follow-through detection with adaptive parameters."""
        if contact is None:
            return None
        
        # Find contact index
        start_idx = 0
        for i, frame in enumerate(data['frames']):
            if frame >= contact:
                start_idx = i
                break
        
        search_window = min(len(data['lead_wrist_x']) - start_idx - 3,
                           max(8, video_length // 5))
        
        if search_window < 6:
            return None
        
        # Multiple follow-through indicators
        indicators = []
        
        # Indicator 1: Hand deceleration
        decel_frame = self._find_hand_deceleration_point(data, start_idx, search_window)
        if decel_frame:
            indicators.append(decel_frame)
        
        # Indicator 2: Body rotation completion
        rotation_frame = self._find_rotation_completion_point(data, start_idx, search_window)
        if rotation_frame:
            indicators.append(rotation_frame)
        
        # Indicator 3: Weight transfer completion
        weight_frame = self._find_final_weight_transfer_point(data, start_idx, search_window)
        if weight_frame:
            indicators.append(weight_frame)
        
        if not indicators:
            return None
        
        # Return median of indicators for robustness
        return int(np.median(indicators))
    
    def _find_hand_deceleration_point(self, data, start_idx, search_window):
        """Find point where hands significantly decelerate."""
        smoothed_x = gaussian_filter1d(data['lead_wrist_x'][start_idx:start_idx+search_window], sigma=1.0)
        smoothed_y = gaussian_filter1d(data['lead_wrist_y'][start_idx:start_idx+search_window], sigma=1.0)
        
        for i in range(4, len(smoothed_x) - 2):
            current_speed = np.sqrt(
                ((smoothed_x[i+2] - smoothed_x[i-2]) / 4)**2 +
                ((smoothed_y[i+2] - smoothed_y[i-2]) / 4)**2
            )
            
            past_speed = np.sqrt(
                ((smoothed_x[i] - smoothed_x[i-4]) / 4)**2 +
                ((smoothed_y[i] - smoothed_y[i-4]) / 4)**2
            )
            
            # Significant deceleration
            if past_speed > 0.06 and current_speed < past_speed * 0.5:
                return data['frames'][start_idx + i]
        
        return None
    
    def _find_rotation_completion_point(self, data, start_idx, search_window):
        """Find point where body rotation completes."""
        # Calculate shoulder rotation
        shoulder_angles = []
        for i in range(start_idx, start_idx + search_window):
            if i >= len(data['lead_shoulder_x']):
                break
            angle = np.arctan2(
                data['back_shoulder_y'][i] - data['lead_shoulder_y'][i],
                data['back_shoulder_x'][i] - data['lead_shoulder_x'][i]
            )
            shoulder_angles.append(angle)
        
        if len(shoulder_angles) < 6:
            return None
        
        smoothed_angles = gaussian_filter1d(shoulder_angles, sigma=1.2)
        
        # Find where rotation velocity drops significantly
        for i in range(3, len(smoothed_angles) - 3):
            rotation_velocity = abs((smoothed_angles[i+2] - smoothed_angles[i-2]) / 4)
            
            if i >= 6:
                past_velocity = abs((smoothed_angles[i] - smoothed_angles[i-6]) / 6)
                if past_velocity > 0.04 and rotation_velocity < past_velocity * 0.3:
                    return data['frames'][start_idx + i]
        
        return None
    
    def _find_final_weight_transfer_point(self, data, start_idx, search_window):
        """Find final weight transfer completion."""
        # Look for final stabilization of weight on front leg
        back_ankle_x = data['back_ankle_x'][start_idx:start_idx+search_window]
        
        if len(back_ankle_x) < 6:
            return None
        
        smoothed_back_ankle = gaussian_filter1d(back_ankle_x, sigma=1.0)
        
        # Find where back foot movement minimizes (final plant)
        min_movement = float('inf')
        min_frame = None
        
        for i in range(3, len(smoothed_back_ankle) - 3):
            movement = abs((smoothed_back_ankle[i+2] - smoothed_back_ankle[i-2]) / 4)
            
            if movement < min_movement:
                min_movement = movement
                min_frame = data['frames'][start_idx + i]
        
        return min_frame if min_movement < 0.02 else None
    
    def calculate_hip_shoulder_separation_robust(self, landmarks_over_time, swing_phases, handedness):
        """Enhanced hip-shoulder separation calculation with improved accuracy."""
        separation_data = []
        is_lefty = handedness == "left"
        
        # Define landmarks based on handedness for consistent measurement
        if is_lefty:
            lead_shoulder_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            back_shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            lead_hip_idx = mp_pose.PoseLandmark.RIGHT_HIP.value
            back_hip_idx = mp_pose.PoseLandmark.LEFT_HIP.value
        else:
            lead_shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            back_shoulder_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            lead_hip_idx = mp_pose.PoseLandmark.LEFT_HIP.value
            back_hip_idx = mp_pose.PoseLandmark.RIGHT_HIP.value
        
        valid_frames = 0
        for frame_idx, lm in enumerate(landmarks_over_time):
            try:
                # Get normalized positions using enhanced normalization
                lead_shoulder_norm = self.normalize_landmark_position(lm[lead_shoulder_idx], lm)
                back_shoulder_norm = self.normalize_landmark_position(lm[back_shoulder_idx], lm)
                lead_hip_norm = self.normalize_landmark_position(lm[lead_hip_idx], lm)
                back_hip_norm = self.normalize_landmark_position(lm[back_hip_idx], lm)
                
                if all([lead_shoulder_norm, back_shoulder_norm, lead_hip_norm, back_hip_norm]):
                    # Calculate angles using normalized coordinates
                    hip_angle = np.degrees(np.arctan2(
                        back_hip_norm['y'] - lead_hip_norm['y'],
                        back_hip_norm['x'] - lead_hip_norm['x']
                    ))
                    
                    shoulder_angle = np.degrees(np.arctan2(
                        back_shoulder_norm['y'] - lead_shoulder_norm['y'],
                        back_shoulder_norm['x'] - lead_shoulder_norm['x']
                    ))
                    
                    # Calculate separation with improved angle handling
                    raw_separation = shoulder_angle - hip_angle
                    # Normalize to -180 to 180 range
                    separation = ((raw_separation + 180) % 360) - 180
                    separation = abs(separation)  # Take absolute value for separation magnitude
                    
                    separation_data.append({
                        'frame': frame_idx,
                        'hip_angle': hip_angle,
                        'shoulder_angle': shoulder_angle,
                        'separation': separation,
                        'body_scale': lead_shoulder_norm.get('scale', 1.0)
                    })
                    valid_frames += 1
            except:
                continue
        
        # Enhanced statistics calculation
        if separation_data and valid_frames > len(landmarks_over_time) * 0.6:
            separations = [d['separation'] for d in separation_data]
            
            # Use smoothed data for more stable max detection
            if len(separations) > 5:
                smoothed_separations = gaussian_filter1d(separations, sigma=1.5)
                max_idx = np.argmax(smoothed_separations)
                max_separation_data = separation_data[max_idx].copy()
                max_separation_data['separation'] = smoothed_separations[max_idx]
            else:
                max_separation_data = max(separation_data, key=lambda x: x['separation'])
            
            # Calculate percentile-based average to reduce outlier influence
            avg_separation = np.mean([s for s in separations if s <= np.percentile(separations, 85)])
            
            return {
                'data': separation_data,
                'max_separation': max_separation_data,
                'average_separation': avg_separation,
                'data_quality': valid_frames / len(landmarks_over_time)
            }
        
        return {
            'data': [],
            'max_separation': None,
            'average_separation': 0.0,
            'data_quality': 0.0
        }

    def calculate_torso_lean_at_frame(self, landmarks, handedness) -> dict:
        """
        Calculates torso lean angle at a given frame using normalized mid-shoulder and mid-hip positions.
        Positive angle = forward lean toward plate; negative = leaning back.
        """
        try:
            # Midpoints between shoulders and hips
            mid_shoulder = np.array([
                (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2,
                (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2
            ])
            mid_hip = np.array([
                (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2,
                (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
            ])

            # Torso vector
            torso_vector = mid_shoulder - mid_hip
            vertical_ref = np.array([0, -1])  # Upward vertical

            # Angle between torso vector and vertical
            dot_product = np.dot(torso_vector, vertical_ref)
            magnitudes = np.linalg.norm(torso_vector) * np.linalg.norm(vertical_ref)
            angle_rad = np.arccos(np.clip(dot_product / magnitudes, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)

            # Determine sign based on forward or backward lean
            forward_lean = (torso_vector[0] > 0) if handedness == "right" else (torso_vector[0] < 0)
            if not forward_lean:
                angle_deg *= -1

            return {
                "torso_lean_angle": angle_deg,
                "forward_lean": forward_lean,
                "torso_vector": torso_vector.tolist()
            }

        except Exception as e:
            return {
                "torso_lean_angle": None,
                "error": str(e)
            }


# Keep the same video loading functions as the original
async def analyze_video_from_url(url: str):
    """
    Main analysis function with enhanced robustness for different video characteristics.
    """
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

    # Initialize the enhanced analyzer
    analyzer = RobustBaseballSwingAnalyzer()
    
    frame_count = 0
    landmarks_over_time = []
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"📹 Processing video: {total_frames} frames at {fps:.1f} FPS ({duration:.1f}s)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = analyzer.pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks_over_time.append(results.pose_landmarks.landmark)

        frame_count += 1

    cap.release()
    os.remove(video_path)

    if len(landmarks_over_time) < 10:
        return [{"message": "Insufficient pose detection. Ensure clear view of full body throughout the swing."}], {}

    print(f"🧠 Analyzing {len(landmarks_over_time)} frames with pose data...")

    # Enhanced handedness detection
    handedness_result = analyzer.detect_handedness_fusion(landmarks_over_time)
    handedness = handedness_result["handedness"]
    
    print(f"🤚 Detected handedness: {handedness} (confidence: {handedness_result['confidence']:.2f})")
    
    # Print debug info for handedness detection
    if "debug" in handedness_result:
        print("   Handedness detection details:")
        if isinstance(handedness_result.get("debug"), dict):
            for method, details in handedness_result["debug"].items():
                print(f"   - {method}: {details}")
        else:
            print(f"   Debug: {handedness_result['debug']}")

    
    # Enhanced swing phase detection
    swing_phases = analyzer.detect_precise_swing_timing(landmarks_over_time, handedness)

    # Log hip-shoulder separation at foot plant if available
    foot_plant_frame = swing_phases.get("foot_plant")
    if foot_plant_frame is not None:
        try:
            lm = landmarks_over_time[foot_plant_frame]
            if handedness == "left":
                lead_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                back_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                lead_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
                back_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
            else:
                lead_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                back_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                lead_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
                back_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]

            hip_angle = np.degrees(np.arctan2(back_hip.y - lead_hip.y, back_hip.x - lead_hip.x))
            shoulder_angle = np.degrees(np.arctan2(back_shoulder.y - lead_shoulder.y, back_shoulder.x - lead_shoulder.x))
            separation = abs(((shoulder_angle - hip_angle + 180) % 360) - 180)  # Normalize and take absolute value

            print(f"📐 Separation at foot plant (Frame {foot_plant_frame}): {separation:.1f}°")

            if separation < 20 or separation > 40:
                feedback.append({
                    "frame": foot_plant_frame,
                    "issue": f"Hip-shoulder separation at foot plant is {separation:.1f}° (ideal: 20–40°)",
                    "suggested_drill": "Improve separation timing and coil mechanics with rotational drills",
                    "severity": "medium"
                })


            torso_lean_result = analyzer.calculate_torso_lean_at_frame(landmarks_over_time[foot_plant_frame], handedness)
            angle = torso_lean_result.get("torso_lean_angle")
            
            if angle is not None:
                direction = "forward" if angle > 0 else "backward"
                print(f"📏 Torso lean at foot plant (Frame {foot_plant_frame}): {angle:.1f}° ({direction})")
            else:
                print("⚠️ Could not calculate torso lean at foot plant.")

        
        except Exception as e:
            print(f"⚠️ Could not compute separation at foot plant: {e}")

    
    print("⏱️ Swing phases detected:")
    for phase, frame in swing_phases.items():
        if frame is not None:
            time_seconds = frame / fps if fps > 0 else frame * 0.033
            print(f"   • {phase.replace('_', ' ').title()}: Frame {frame} ({time_seconds:.2f}s)")
        else:
            print(f"   • {phase.replace('_', ' ').title()}: Not detected")

    # Enhanced hip-shoulder separation analysis
    separation_analysis = analyzer.calculate_hip_shoulder_separation_robust(
        landmarks_over_time, swing_phases, handedness
    )

    # Generate enhanced feedback based on analysis
    if separation_analysis['max_separation'] and separation_analysis['data_quality'] > 0.6:
        max_sep = separation_analysis['max_separation']['separation']
        avg_sep = separation_analysis['average_separation']
        
        if max_sep < 15:
            feedback.append({
                "frame": separation_analysis['max_separation']['frame'],
                "issue": f"Low hip-shoulder separation ({max_sep:.1f}°)",
                "suggested_drill": "Practice coil drills and hip turn exercises to improve torque generation",
                "severity": "high"
            })
        elif max_sep < 25:
            feedback.append({
                "frame": separation_analysis['max_separation']['frame'],
                "issue": f"Below optimal hip-shoulder separation ({max_sep:.1f}°)",
                "suggested_drill": "Work on timing hip initiation before shoulder turn",
                "severity": "medium"
            })
        elif max_sep > 55:
            feedback.append({
                "frame": separation_analysis['max_separation']['frame'],
                "issue": f"Very high separation ({max_sep:.1f}°) - check timing coordination",
                "suggested_drill": "Practice smooth kinetic chain drills to improve timing",
                "severity": "medium"
            })
        
        print(f"🔄 Hip-shoulder separation: Peak {max_sep:.1f}° (avg: {avg_sep:.1f}°)")
    else:
        feedback.append({
            "frame": "overall",
            "issue": "Hip-shoulder separation could not be measured reliably",
            "suggested_drill": "Ensure clear side view of the swing with full body visible",
            "severity": "low"
        })

    # Enhanced swing phase timing analysis
    detected_phases = [phase for phase, frame in swing_phases.items() if frame is not None]
    if len(detected_phases) < 3:
        feedback.append({
            "frame": "overall",
            "issue": f"Only {len(detected_phases)} of 5 swing phases detected clearly",
            "suggested_drill": "Ensure video captures complete swing from stance through follow-through",
            "severity": "medium"
        })
    
    # Phase timing validation
    if swing_phases.get("swing_start") and swing_phases.get("contact"):
        swing_duration = swing_phases["contact"] - swing_phases["swing_start"]
        if swing_duration < 8:
            feedback.append({
                "frame": swing_phases["swing_start"],
                "issue": f"Very quick swing ({swing_duration} frames)",
                "suggested_drill": "Consider working on tempo and timing control",
                "severity": "low"
            })

    # Handedness confidence analysis
    if handedness == "unknown":
        feedback.append({
            "frame": "setup",
            "issue": "Could not determine handedness reliably",
            "suggested_drill": "Ensure clear side view showing batting stance and initial setup",
            "severity": "high"
        })
    elif handedness_result["confidence"] < 0.7:
        feedback.append({
            "frame": "setup", 
            "issue": f"Handedness detection has moderate confidence ({handedness_result['confidence']:.2f})",
            "suggested_drill": "Verify analysis results - ensure clear side view of batting stance",
            "severity": "low"
        })

    # Data quality assessment
    data_quality = separation_analysis.get('data_quality', 0)
    if data_quality < 0.7:
        feedback.append({
            "frame": "overall",
            "issue": f"Pose detection quality: {data_quality*100:.0f}%",
            "suggested_drill": "Improve lighting and ensure full body is visible throughout swing",
            "severity": "medium" if data_quality < 0.5 else "low"
        })

    biomarker_results = {
        "handedness": handedness_result,
        "swing_phases": swing_phases,
        "separation_analysis": separation_analysis,
        "video_info": {
            "fps": fps,
            "frames_analyzed": len(landmarks_over_time),
            "duration_seconds": duration,
            "data_quality": data_quality
        }
    }

    print("✅ Enhanced analysis complete!")
    return feedback or [{"message": "No major issues detected. Swing mechanics appear solid!"}], biomarker_results


async def generate_annotated_video(video_url: str, swing_phases: dict) -> str:
    """Generates an annotated swing video with phase labels and joint overlays."""
    print(f"🎞️ Generating annotated video for: {video_url}")

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
    output_path = "annotated_output.mp4"
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

            # ✅ Draw hip–shoulder separation angle at foot plant
            if frame_idx == swing_phases.get("foot_plant"):
                landmarks = results.pose_landmarks.landmark

                # Get shoulder and hip points
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

                # Convert normalized coordinates to pixel values
                def denorm(point):
                    return int(point.x * width), int(point.y * height)

                ls = denorm(left_shoulder)
                rs = denorm(right_shoulder)
                lh = denorm(left_hip)
                rh = denorm(right_hip)

                # Draw lines
                cv2.line(frame, ls, rs, (0, 255, 255), 2)  # yellow = shoulders
                cv2.line(frame, lh, rh, (255, 0, 255), 2)  # magenta = hips

                # Calculate angles
                shoulder_angle = np.degrees(np.arctan2(rs[1] - ls[1], rs[0] - ls[0]))
                hip_angle = np.degrees(np.arctan2(rh[1] - lh[1], rh[0] - lh[0]))
                separation = shoulder_angle - hip_angle
                separation = ((separation + 180) % 360) - 180  # Normalize to -180 to 180
                separation = abs(separation)

                # Display angle on screen
                cv2.putText(
                    frame,
                    f"Hip-Shoulder Sep: {separation:.1f} degrees",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 0),
                    2
                )


        # Enhanced phase labeling with colors
        phase_colors = {
            "stride_start": (0, 255, 255),    # Yellow
            "foot_plant": (255, 165, 0),      # Orange  
            "swing_start": (0, 255, 0),       # Green
            "contact": (255, 0, 0),           # Red
            "follow_through": (255, 0, 255)   # Magenta
        }
        
        current_phase = None
        for phase_name, phase_frame in swing_phases.items():
            if phase_frame == frame_idx:
                current_phase = phase_name
                break
        
        if current_phase:
            label = current_phase.replace("_", " ").title()
            color = phase_colors.get(current_phase, (255, 255, 255))
            cv2.putText(frame, f"{label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"{label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1)

        cv2.putText(frame, f"Frame {frame_idx}", (30, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    os.remove(input_video_path)

    print(f"✅ Enhanced annotated video saved to: {output_path}")
    return output_path



