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
        phases["swing_start"] = self._detect_swing_start_adaptive(movement_data, phases, video_length)
        phases["contact"] = self._detect_contact_adaptive(movement_data, phases["swing_start"], video_length)
        phases["follow_through"] = self._detect_follow_through_adaptive(movement_data, phases["contact"], video_length)
        
        return phases
    
    def _extract_adaptive_movement_data(self, landmarks_over_time, is_lefty, video_length):
        """Extract movement data with adaptive sampling based on video characteristics."""
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
            'lead_knee_x': [], 'lead_knee_y': [],
            'back_knee_x': [], 'back_knee_y': [],
            'lead_hip_x': [], 'lead_hip_y': [],
            'back_hip_x': [], 'back_hip_y': [],
            'lead_shoulder_x': [], 'lead_shoulder_y': [],
            'back_shoulder_x': [], 'back_shoulder_y': [],
            'lead_wrist_x': [], 'lead_wrist_y': [],
            'back_wrist_x': [], 'back_wrist_y': [],
            'lead_elbow_x': [], 'lead_elbow_y': [],
            'back_elbow_x': [], 'back_elbow_y': [],
            # Additional tracking for better analysis
            'body_scales': [],  # Track zoom level changes
            'hip_center_x': [], 'hip_center_y': [],
        }
        
        landmark_indices = {
            'lead_ankle': getattr(mp_pose.PoseLandmark, f'{lead_side}_ANKLE').value,
            'back_ankle': getattr(mp_pose.PoseLandmark, f'{back_side}_ANKLE').value,
            'lead_knee': getattr(mp_pose.PoseLandmark, f'{lead_side}_KNEE').value,
            'back_knee': getattr(mp_pose.PoseLandmark, f'{back_side}_KNEE').value,
            'lead_hip': getattr(mp_pose.PoseLandmark, f'{lead_side}_HIP').value,
            'back_hip': getattr(mp_pose.PoseLandmark, f'{back_side}_HIP').value,
            'lead_shoulder': getattr(mp_pose.PoseLandmark, f'{lead_side}_SHOULDER').value,
            'back_shoulder': getattr(mp_pose.PoseLandmark, f'{back_side}_SHOULDER').value,
            'lead_wrist': getattr(mp_pose.PoseLandmark, f'{lead_side}_WRIST').value,
            'back_wrist': getattr(mp_pose.PoseLandmark, f'{back_side}_WRIST').value,
            'lead_elbow': getattr(mp_pose.PoseLandmark, f'{lead_side}_ELBOW').value,
            'back_elbow': getattr(mp_pose.PoseLandmark, f'{back_side}_ELBOW').value,
        }
        
        for frame_idx, landmarks in enumerate(landmarks_over_time):
            try:
                frame_data = {}
                all_present = True
                
                # Get body metrics for this frame
                body_metrics = self.calculate_adaptive_body_metrics(landmarks)
                if not body_metrics or not body_metrics['measurements_valid']:
                    continue
                
                # Store body scale and hip center for tracking zoom/movement
                data['body_scales'].append(body_metrics['scale'])
                data['hip_center_x'].append(body_metrics['hip_center'][0])
                data['hip_center_y'].append(body_metrics['hip_center'][1])
                
                # Get normalized positions for all landmarks
                for part_name, landmark_idx in landmark_indices.items():
                    normalized = self.normalize_landmark_position(landmarks[landmark_idx], landmarks)
                    if normalized:
                        frame_data[f'{part_name}_x'] = normalized['x']
                        frame_data[f'{part_name}_y'] = normalized['y']
                    else:
                        all_present = False
                        break
                
                if all_present:
                    data['frames'].append(frame_idx)
                    for key, value in frame_data.items():
                        data[key].append(value)
                        
            except:
                continue
        
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

        print(f"🦶 Convergence threshold: {convergence_threshold:.4f}")
        print(f"🕒 Stability: {min_stability_frames}-{max_stability_frames} frames | σ: {smoothing_sigma:.2f} | FPS: {fps:.2f}")

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



    def _detect_swing_start_adaptive(self, data, phases, video_length):
        """Enhanced swing start detection using multiple biomechanical indicators."""
        # Determine search start based on available phases
        search_start_idx = 0
        if phases.get("foot_plant"):
            for i, frame in enumerate(data['frames']):
                if frame >= phases["foot_plant"]:
                    search_start_idx = max(0, i - 2)
                    break
        elif phases.get("stride_start"):
            for i, frame in enumerate(data['frames']):
                if frame >= phases["stride_start"]:
                    search_start_idx = max(0, i + 2)  # Swing typically starts after stride
                    break
        
        if len(data['lead_wrist_x']) < search_start_idx + 8:
            return None
        
        # Adaptive search window
        search_window = min(len(data['frames']) - search_start_idx - 3,
                           max(10, video_length // 3))
        
        # Multiple swing initiation indicators
        indicators = []
        
        # Indicator 1: Hip rotation acceleration
        hip_indicator = self._detect_hip_rotation_start_adaptive(data, search_start_idx, search_window)
        if hip_indicator:
            indicators.append(("hip_rotation", hip_indicator, 3))  # High weight
        
        # Indicator 2: Hand/bat acceleration
        hand_indicator = self._detect_hand_acceleration_start_adaptive(data, search_start_idx, search_window)
        if hand_indicator:
            indicators.append(("hand_acceleration", hand_indicator, 3))  # High weight
        
        # Indicator 3: Shoulder sequence initiation
        shoulder_indicator = self._detect_shoulder_sequence_start_adaptive(data, search_start_idx, search_window)
        if shoulder_indicator:
            indicators.append(("shoulder_sequence", shoulder_indicator, 2))  # Medium weight
        
        # Indicator 4: Weight transfer detection
        weight_indicator = self._detect_weight_transfer_start_adaptive(data, search_start_idx, search_window)
        if weight_indicator:
            indicators.append(("weight_transfer", weight_indicator, 2))  # Medium weight
        
        # Indicator 5: Kinetic chain initiation
        kinetic_indicator = self._detect_kinetic_chain_start_adaptive(data, search_start_idx, search_window)
        if kinetic_indicator:
            indicators.append(("kinetic_chain", kinetic_indicator, 2))  # Medium weight
        
        if not indicators:
            return None
        
        # Weighted consensus finding
        frame_weights = {}
        for indicator_name, frame, weight in indicators:
            for check_frame in range(frame - 2, frame + 3):
                if check_frame not in frame_weights:
                    frame_weights[check_frame] = 0
                frame_weights[check_frame] += weight * (3 - abs(check_frame - frame)) / 3  # Distance decay
        
        # Find frame with highest weighted support
        if frame_weights:
            best_frame = max(frame_weights.items(), key=lambda x: x[1])
            if best_frame[1] >= 4:  # Minimum weighted threshold
                return best_frame[0]
        
        # Fallback: use earliest high-confidence indicator
        high_confidence_indicators = [(name, frame) for name, frame, weight in indicators if weight >= 3]
        if high_confidence_indicators:
            return min(frame for _, frame in high_confidence_indicators)
        
        return min(frame for _, frame, _ in indicators) if indicators else None
    
    def _detect_hip_rotation_start_adaptive(self, data, start_idx, search_window):
        """Adaptive hip rotation detection."""
        end_idx = min(start_idx + search_window, len(data['lead_hip_x']))
        
        if end_idx - start_idx < 6:
            return None
        
        # Calculate hip angles with enhanced smoothing
        hip_angles = []
        for i in range(start_idx, end_idx):
            try:
                angle = np.degrees(np.arctan2(
                    data['back_hip_y'][i] - data['lead_hip_y'][i],
                    data['back_hip_x'][i] - data['lead_hip_x'][i]
                ))
                hip_angles.append(angle)
            except:
                hip_angles.append(hip_angles[-1] if hip_angles else 0)
        
        if len(hip_angles) < 6:
            return None
        
        # Smooth angles and calculate angular velocity and acceleration
        smoothed_angles = gaussian_filter1d(hip_angles, sigma=1.2)
        
        for i in range(3, len(smoothed_angles) - 3):
            # Angular velocity (degrees per frame)
            angular_velocity = (smoothed_angles[i+2] - smoothed_angles[i-2]) / 4
            # Angular acceleration  
            angular_acceleration = (smoothed_angles[i+3] - 2*smoothed_angles[i] + smoothed_angles[i-3]) / 9
            
            # Look for significant rotational initiation
            if abs(angular_velocity) > 2.5 and abs(angular_acceleration) > 1.0:
                return data['frames'][start_idx + i]
        
        return None
    
    def _detect_hand_acceleration_start_adaptive(self, data, start_idx, search_window):
        """Adaptive hand acceleration detection with both hands."""
        end_idx = min(start_idx + search_window, len(data['lead_wrist_x']))
        
        if end_idx - start_idx < 6:
            return None
        
        # Enhanced smoothing for hand tracking
        smoothing_sigma = 1.2
        lead_x_smooth = gaussian_filter1d(data['lead_wrist_x'][start_idx:end_idx], sigma=smoothing_sigma)
        lead_y_smooth = gaussian_filter1d(data['lead_wrist_y'][start_idx:end_idx], sigma=smoothing_sigma)
        back_x_smooth = gaussian_filter1d(data['back_wrist_x'][start_idx:end_idx], sigma=smoothing_sigma)
        back_y_smooth = gaussian_filter1d(data['back_wrist_y'][start_idx:end_idx], sigma=smoothing_sigma)
        
        for i in range(3, len(lead_x_smooth) - 3):
            # Calculate velocities for both hands
            lead_vel_x = (lead_x_smooth[i+2] - lead_x_smooth[i-2]) / 4
            lead_vel_y = (lead_y_smooth[i+2] - lead_y_smooth[i-2]) / 4
            back_vel_x = (back_x_smooth[i+2] - back_x_smooth[i-2]) / 4
            back_vel_y = (back_y_smooth[i+2] - back_y_smooth[i-2]) / 4
            
            lead_speed = np.sqrt(lead_vel_x**2 + lead_vel_y**2)
            back_speed = np.sqrt(back_vel_x**2 + back_vel_y**2)
            
            # Calculate accelerations
            if i >= 4:
                prev_lead_vel_x = (lead_x_smooth[i] - lead_x_smooth[i-4]) / 4
                prev_lead_vel_y = (lead_y_smooth[i] - lead_y_smooth[i-4]) / 4
                prev_lead_speed = np.sqrt(prev_lead_vel_x**2 + prev_lead_vel_y**2)
                
                lead_acceleration = lead_speed - prev_lead_speed
                
                # Swing start: significant acceleration in both hands
                if lead_speed > 0.06 and back_speed > 0.04 and lead_acceleration > 0.02:
                    return data['frames'][start_idx + i]
        
        return None
    
    def _detect_shoulder_sequence_start_adaptive(self, data, start_idx, search_window):
        """Adaptive shoulder sequence detection."""
        end_idx = min(start_idx + search_window, len(data['back_shoulder_x']))
        
        if end_idx - start_idx < 6:
            return None
        
        # Track back shoulder movement (key swing initiator)
        back_shoulder_x = gaussian_filter1d(data['back_shoulder_x'][start_idx:end_idx], sigma=1.0)
        
        # Also track lead shoulder for sequence timing
        lead_shoulder_x = gaussian_filter1d(data['lead_shoulder_x'][start_idx:end_idx], sigma=1.0)
        
        for i in range(3, len(back_shoulder_x) - 3):
            # Back shoulder velocity (forward movement)
            back_shoulder_vel = (back_shoulder_x[i+3] - back_shoulder_x[i-3]) / 6
            
            # Lead shoulder velocity (should be less initially)
            lead_shoulder_vel = (lead_shoulder_x[i+3] - lead_shoulder_x[i-3]) / 6
            
            # Shoulder separation rate
            separation_rate = abs(back_shoulder_vel - lead_shoulder_vel)
            
            # Shoulder sequence: back shoulder initiates with lead shoulder following
            if abs(back_shoulder_vel) > 0.035 and separation_rate > 0.025:
                return data['frames'][start_idx + i]
        
        return None
    
    def _detect_weight_transfer_start_adaptive(self, data, start_idx, search_window):
        """Adaptive weight transfer detection."""
        end_idx = min(start_idx + search_window, len(data['back_knee_x']))
        
        if end_idx - start_idx < 6:
            return None
        
        # Track knee and hip positions for weight transfer
        lead_knee_x = gaussian_filter1d(data['lead_knee_x'][start_idx:end_idx], sigma=1.0)
        back_knee_x = gaussian_filter1d(data['back_knee_x'][start_idx:end_idx], sigma=1.0)
        lead_hip_x = gaussian_filter1d(data['lead_hip_x'][start_idx:end_idx], sigma=1.0)
        back_hip_x = gaussian_filter1d(data['back_hip_x'][start_idx:end_idx], sigma=1.0)
        
        for i in range(3, len(lead_knee_x) - 3):
            # Calculate weight shift indicators
            knee_separation = abs(lead_knee_x[i] - back_knee_x[i])
            hip_separation = abs(lead_hip_x[i] - back_hip_x[i])
            
            # Rate of change in separations
            if i >= 4:
                prev_knee_sep = abs(lead_knee_x[i-4] - back_knee_x[i-4])
                prev_hip_sep = abs(lead_hip_x[i-4] - back_hip_x[i-4])
                
                knee_sep_rate = abs(knee_separation - prev_knee_sep)
                hip_sep_rate = abs(hip_separation - prev_hip_sep)
                
                # Weight transfer: changing separation patterns
                if knee_sep_rate > 0.025 or hip_sep_rate > 0.02:
                    return data['frames'][start_idx + i]
        
        return None
    
    def _detect_kinetic_chain_start_adaptive(self, data, start_idx, search_window):
        """Detect start of kinetic chain sequence (hips->shoulders->arms)."""
        end_idx = min(start_idx + search_window, len(data['lead_hip_x']))
        
        if end_idx - start_idx < 8:
            return None
        
        # Calculate center of mass movement (simplified)
        com_x = []
        com_y = []
        
        for i in range(start_idx, end_idx):
            # Weighted center of mass approximation
            com_x_frame = (data['lead_hip_x'][i] + data['back_hip_x'][i] + 
                          data['lead_shoulder_x'][i] + data['back_shoulder_x'][i]) / 4
            com_y_frame = (data['lead_hip_y'][i] + data['back_hip_y'][i] +
                          data['lead_shoulder_y'][i] + data['back_shoulder_y'][i]) / 4
            com_x.append(com_x_frame)
            com_y.append(com_y_frame)
        
        com_x_smooth = gaussian_filter1d(com_x, sigma=1.0)
        
        # Look for center of mass acceleration (kinetic chain initiation)
        for i in range(3, len(com_x_smooth) - 3):
            com_velocity = (com_x_smooth[i+2] - com_x_smooth[i-2]) / 4
            
            if i >= 4:
                prev_com_velocity = (com_x_smooth[i] - com_x_smooth[i-4]) / 4
                com_acceleration = com_velocity - prev_com_velocity
                
                # Kinetic chain starts with center of mass acceleration
                if abs(com_acceleration) > 0.015 and abs(com_velocity) > 0.02:
                    return data['frames'][start_idx + i]
        
        return None
    
    def _detect_contact_adaptive(self, data, swing_start, video_length):
        """Enhanced contact detection using multiple adaptive indicators."""
        if swing_start is None:
            return None
        
        # Find swing start index
        start_idx = 0
        for i, frame in enumerate(data['frames']):
            if frame >= swing_start:
                start_idx = i
                break
        
        search_window = min(len(data['lead_wrist_x']) - start_idx - 3,
                           max(12, video_length // 4))
        
        if search_window < 8:
            return None
        
        # Multiple contact indicators with adaptive weights
        contact_indicators = []
        
        # Indicator 1: Maximum hand separation (classic method, enhanced)
        max_sep_frame = self._find_maximum_hand_separation_adaptive(data, start_idx, search_window)
        if max_sep_frame:
            contact_indicators.append(("max_separation", max_sep_frame, 3))
        
        # Indicator 2: Peak bat speed
        max_speed_frame = self._find_maximum_bat_speed_adaptive(data, start_idx, search_window)
        if max_speed_frame:
            contact_indicators.append(("max_speed", max_speed_frame, 3))
        
        # Indicator 3: Lead arm extension peak
        extension_frame = self._find_lead_arm_extension_max_adaptive(data, start_idx, search_window)
        if extension_frame:
            contact_indicators.append(("arm_extension", extension_frame, 2))
        
        # Indicator 4: Hip-shoulder sequence peak
        sequence_frame = self._find_kinetic_sequence_peak_adaptive(data, start_idx, search_window)
        if sequence_frame:
            contact_indicators.append(("sequence_peak", sequence_frame, 2))
        
        # Indicator 5: Weight transfer completion
        weight_completion_frame = self._find_weight_transfer_completion_adaptive(data, start_idx, search_window)
        if weight_completion_frame:
            contact_indicators.append(("weight_completion", weight_completion_frame, 1))
        
        if not contact_indicators:
            return None
        
        # Weighted consensus with temporal clustering
        frame_weights = {}
        for indicator_name, frame, weight in contact_indicators:
            for check_frame in range(frame - 3, frame + 4):
                if check_frame not in frame_weights:
                    frame_weights[check_frame] = 0
                # Distance-weighted scoring
                distance_weight = (4 - abs(check_frame - frame)) / 4
                frame_weights[check_frame] += weight * distance_weight
        
        if frame_weights:
            best_frame = max(frame_weights.items(), key=lambda x: x[1])
            if best_frame[1] >= 5:  # Minimum weighted evidence threshold
                return best_frame[0]
        
        # Fallback to highest confidence single indicator
        high_confidence = [(name, frame) for name, frame, weight in contact_indicators if weight >= 3]
        if high_confidence:
            return min(frame for _, frame in high_confidence)
        
        return None
    
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



