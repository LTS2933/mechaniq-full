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
import uuid

# NEW: supabase-py client
from supabase import create_client, Client


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
        print("🔍 Starting handedness detection...")
        
        if len(landmarks_over_time) < 10:
            print("❌ Insufficient frames for handedness detection")
            return {"handedness": "unknown", "confidence": 0.0, "debug": "Insufficient frames"}

        max_frames = min(40, len(landmarks_over_time))
        direction_scores = []
        valid_frames = 0
        low_visibility_frames = 0
        
        print(f"📊 Analyzing {max_frames} frames for face direction...")

        for i in range(max_frames):
            lm = landmarks_over_time[i]

            try:
                nose = lm[mp_pose.PoseLandmark.NOSE.value]
                leye = lm[mp_pose.PoseLandmark.LEFT_EYE.value]
                reye = lm[mp_pose.PoseLandmark.RIGHT_EYE.value]
                lear = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
                rear = lm[mp_pose.PoseLandmark.RIGHT_EAR.value]

                # Check visibility
                min_visibility = min(nose.visibility, leye.visibility, reye.visibility, lear.visibility, rear.visibility)
                
                if min_visibility < 0.5:
                    low_visibility_frames += 1
                    if i < 10:  # Debug first 10 frames
                        print(f"   Frame {i}: Low visibility ({min_visibility:.2f}) - skipping")
                    continue

                # Calculate distances from nose to eyes/ears
                dist_nose_to_leye = nose.x - leye.x
                dist_nose_to_reye = reye.x - nose.x
                dist_nose_to_lear = nose.x - lear.x
                dist_nose_to_rear = rear.x - nose.x

                # Individual components
                eye_component = dist_nose_to_reye - dist_nose_to_leye
                ear_component = dist_nose_to_rear - dist_nose_to_lear
                
                # Combine signals: negative means facing right (right-handed), positive means facing left (left-handed)
                direction_score = eye_component + ear_component
                direction_scores.append(direction_score)
                valid_frames += 1
                
                # Debug output for first few valid frames
                if valid_frames <= 5:
                    face_direction = "LEFT" if direction_score > 0 else "RIGHT"
                    """
                    print(f"   Frame {i}: score={direction_score:.4f} → facing {face_direction}")
                    print(f"      Eye component: {eye_component:.4f} (R-eye dist: {dist_nose_to_reye:.3f}, L-eye dist: {dist_nose_to_leye:.3f})")
                    print(f"      Ear component: {ear_component:.4f} (R-ear dist: {dist_nose_to_rear:.3f}, L-ear dist: {dist_nose_to_lear:.3f})")
                    """

            except Exception as e:
                if i < 10:  # Debug first 10 frames
                    print(f"   Frame {i}: Error processing landmarks - {e}")
                continue

        #print(f"📈 Processing results: {valid_frames} valid frames, {low_visibility_frames} low visibility frames")

        if len(direction_scores) < 5:
            print("❌ Too few valid face frames for reliable detection")
            return {"handedness": "unknown", "confidence": 0.0, "debug": "Too few valid face frames"}

        # Calculate statistics
        avg_score = sum(direction_scores) / len(direction_scores)
        score_std = np.std(direction_scores) if len(direction_scores) > 1 else 0.0
        positive_scores = sum(1 for s in direction_scores if s > 0)
        negative_scores = sum(1 for s in direction_scores if s < 0)
        
        """
        print(f"📊 Score Analysis:")
        print(f"   Average score: {avg_score:.4f}")
        print(f"   Score std dev: {score_std:.4f}")
        print(f"   Positive scores (facing left): {positive_scores}/{len(direction_scores)} ({positive_scores/len(direction_scores)*100:.1f}%)")
        print(f"   Negative scores (facing right): {negative_scores}/{len(direction_scores)} ({negative_scores/len(direction_scores)*100:.1f}%)")
        print(f"   Score range: [{min(direction_scores):.4f}, {max(direction_scores):.4f}]")
        """

        if abs(avg_score) < 0.01:
            print("❌ Face direction too ambiguous (average score near zero)")
            print("🔄 Attempting hand-shoulder proximity backup detection...")
            return self._detect_handedness_by_hand_position(landmarks_over_time)

        # Determine handedness
        if avg_score > 0:
            handedness = "left"
            reasoning = f"Face pointing LEFT (avg score: {avg_score:.4f} > 0) → LEFT-handed batter"
        else:
            handedness = "right"  
            reasoning = f"Face pointing RIGHT (avg score: {avg_score:.4f} < 0) → RIGHT-handed batter"
        
        confidence = round(min(abs(avg_score) * 10, 1.0), 2)
        
        # Additional confidence factors
        consistency = (max(positive_scores, negative_scores) / len(direction_scores))
        if consistency >= 0.8:
            confidence_level = "VERY HIGH"
        elif consistency >= 0.7:
            confidence_level = "HIGH" 
        elif consistency >= 0.6:
            confidence_level = "MODERATE"
        else:
            confidence_level = "LOW"
        
        # Check face detection confidence - if not 100% consistent, use backup method
        face_confidence_threshold = 0.98  # 95% of frames must agree
        if consistency < face_confidence_threshold:
            print(f"⚠️ Face detection consistency ({consistency:.1%}) below threshold ({face_confidence_threshold:.1%})")
            print("🔄 Using hand-shoulder proximity as backup verification...")
            
            backup_result = self._detect_handedness_by_hand_position(landmarks_over_time)
            
            if backup_result["handedness"] != "unknown":
                if backup_result["handedness"] == handedness:
                    print(f"✅ Hand-shoulder method CONFIRMS face detection: {handedness}")
                    confidence = min(1.0, confidence + 0.1)  # Boost confidence slightly
                else:
                    print(f"🚨 Hand-shoulder method CONTRADICTS face detection!")
                    print(f"   Face method: {handedness} (confidence: {confidence})")
                    print(f"   Hand method: {backup_result['handedness']} (confidence: {backup_result['confidence']})")
                    
                    # Use the method with higher confidence
                    if backup_result["confidence"] > confidence:
                        print(f"📊 Using hand-shoulder result due to higher confidence")
                        return backup_result
                    else:
                        print(f"📊 Keeping face detection result due to higher confidence")
        
        """
        print(f"✅ {confidence_level} CONFIDENCE Detection:")
        print(f"   {reasoning}")
        print(f"   Consistency: {consistency:.1%} of frames agree")
        print(f"   Final confidence score: {confidence}")

        """

        return {
            "handedness": handedness,
            "confidence": confidence,
            "debug": {
                "method": "face direction via relative eye/ear distances",
                "avg_direction_score": round(avg_score, 4),
                "frames_used": len(direction_scores),
                "score_std": round(score_std, 4),
                "consistency": round(consistency, 3),
                "positive_frames": positive_scores,
                "negative_frames": negative_scores,
                "reasoning": reasoning,
                "confidence_level": confidence_level
            }
        }

    def _detect_handedness_by_hand_position(self, landmarks_over_time):
        """
        Backup handedness detection using hand-shoulder proximity.
        Left-handed: hands closer to left shoulder
        Right-handed: hands closer to right shoulder
        """
        #print("🤲 Starting hand-shoulder proximity analysis...")
        
        valid_measurements = []
        
        for i, lm in enumerate(landmarks_over_time[:30]):  # Check first 30 frames
            try:
                # Get shoulder positions
                left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                
                # Get hand positions (wrists as proxy for hands)
                left_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
                right_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                
                # Check visibility
                min_visibility = min(left_shoulder.visibility, right_shoulder.visibility, 
                                    left_wrist.visibility, right_wrist.visibility)
                
                if min_visibility < 0.5:
                    continue
                
                # Calculate hand center
                hand_center_x = (left_wrist.x + right_wrist.x) / 2
                hand_center_y = (left_wrist.y + right_wrist.y) / 2
                
                # Calculate distances from hand center to each shoulder
                dist_to_left_shoulder = np.sqrt(
                    (hand_center_x - left_shoulder.x)**2 + 
                    (hand_center_y - left_shoulder.y)**2
                )
                
                dist_to_right_shoulder = np.sqrt(
                    (hand_center_x - right_shoulder.x)**2 + 
                    (hand_center_y - right_shoulder.y)**2
                )
                
                # Score: negative = closer to left shoulder (lefty), positive = closer to right shoulder (righty)
                proximity_score = dist_to_left_shoulder - dist_to_right_shoulder
                
                valid_measurements.append(proximity_score)
                
                # Debug first few measurements
                if len(valid_measurements) <= 3:
                    closer_to = "LEFT shoulder" if proximity_score < 0 else "RIGHT shoulder"
                    suggested_handedness = "LEFT-handed" if proximity_score < 0 else "RIGHT-handed"
                    print(f"   Frame {i}: hands closer to {closer_to} → suggests {suggested_handedness}")
                    print(f"      Distance to L-shoulder: {dist_to_left_shoulder:.3f}")
                    print(f"      Distance to R-shoulder: {dist_to_right_shoulder:.3f}")
                    print(f"      Proximity score: {proximity_score:.4f}")
                
            except Exception as e:
                continue
        
        if len(valid_measurements) < 5:
            print("❌ Not enough valid hand-shoulder measurements")
            return {"handedness": "unknown", "confidence": 0.0, "debug": "Insufficient hand-shoulder data"}
        
        # Analyze results
        avg_proximity = np.mean(valid_measurements)
        left_count = sum(1 for score in valid_measurements if score < 0)
        right_count = sum(1 for score in valid_measurements if score > 0)
        consistency = max(left_count, right_count) / len(valid_measurements)
        
        """
        print(f"📊 Hand-Shoulder Analysis:")
        print(f"   Average proximity score: {avg_proximity:.4f}")
        print(f"   Closer to left shoulder: {left_count}/{len(valid_measurements)} ({left_count/len(valid_measurements)*100:.1f}%)")
        print(f"   Closer to right shoulder: {right_count}/{len(valid_measurements)} ({right_count/len(valid_measurements)*100:.1f}%)")
        print(f"   Consistency: {consistency:.1%}")
        """
        
        if abs(avg_proximity) < 0.01:
            print("❌ Hand position too ambiguous (equal distance to both shoulders)")
            return {"handedness": "unknown", "confidence": 0.0, "debug": "Hands equidistant from shoulders"}
        
        # Determine handedness
        if avg_proximity < 0:
            handedness = "left"
            reasoning = f"Hands closer to LEFT shoulder (avg score: {avg_proximity:.4f}) → LEFT-handed"
        else:
            handedness = "right"
            reasoning = f"Hands closer to RIGHT shoulder (avg score: {avg_proximity:.4f}) → RIGHT-handed"
        
        confidence = min(abs(avg_proximity) * 5 + consistency * 0.3, 1.0)  # Scale differently than face detection
        
        print(f"✅ Hand-shoulder detection result:")
        print(f"   {reasoning}")
        print(f"   Confidence: {confidence:.2f}")
        
        return {
            "handedness": handedness,
            "confidence": confidence,
            "debug": {
                "method": "hand-shoulder proximity analysis",
                "avg_proximity_score": round(avg_proximity, 4),
                "measurements_used": len(valid_measurements),
                "consistency": round(consistency, 3),
                "reasoning": reasoning
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
            # NEW: foot landmarks for robust foot plant detection
            'lead_heel_x': [], 'lead_heel_y': [],
            'back_heel_x': [], 'back_heel_y': [],
            'lead_foot_index_x': [], 'lead_foot_index_y': [],
            'back_foot_index_x': [], 'back_foot_index_y': [],
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
            # Add foot landmarks
            'lead_heel':    getattr(mp_pose.PoseLandmark, f'{lead_side}_HEEL').value,
            'back_heel':    getattr(mp_pose.PoseLandmark, f'{back_side}_HEEL').value,
            'lead_foot_index': getattr(mp_pose.PoseLandmark, f'{lead_side}_FOOT_INDEX').value,
            'back_foot_index': getattr(mp_pose.PoseLandmark, f'{back_side}_FOOT_INDEX').value,
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
    

    # --- Rewritten Foot Plant Detection ---
    def _detect_foot_plant_adaptive(self, data, stride_start, video_length):
        """
        Detect foot plant by checking when at least 2 of 3 lead foot landmarks 
        (ankle, heel, foot_index) align in Y-position with the corresponding back foot landmarks.
        Returns immediately when a frame meets the 2/3 requirement.
        """
        if stride_start is None or len(data['lead_ankle_y']) < 10:
            print("🚫 Not enough data to detect foot plant.")
            return None

        try:
            stride_index = next(i for i, f in enumerate(data['frames']) if f >= stride_start)
        except StopIteration:
            print("⚠️ Could not find stride start in frame list.")
            return None

        fps = max(len(data['frames']) / video_length if video_length > 0 else 30, 10)
        search_start = stride_index + max(5, int(0.3 * fps))
        search_end = min(len(data['frames']) - 5, stride_index + int(5.0 * fps))

        #print(f"🔍 Looking for foot plant from frame {data['frames'][search_start]} to {data['frames'][search_end]}")

        tolerance = 0.04
        required_matches = 2  # at least 2 of 3 landmarks must match

        for i in range(search_start, search_end):
            current_frame = data['frames'][i]
            
            ankle_diff = data['lead_ankle_y'][i] - data['back_ankle_y'][i]
            heel_diff = data['lead_heel_y'][i] - data['back_heel_y'][i]
            foot_diff = data['lead_foot_index_y'][i] - data['back_foot_index_y'][i]

            diffs = [ankle_diff, heel_diff, foot_diff]
            num_pass = sum(abs(d) <= tolerance for d in diffs)

            #print(f"🧪 Frame {current_frame}: ankle={ankle_diff:.4f}, heel={heel_diff:.4f}, foot={foot_diff:.4f} → {num_pass}/3 pass")

            if num_pass >= required_matches:
                print(f"✅ Foot plant detected at frame {current_frame} ({num_pass}/3 landmarks aligned)")
                return current_frame

        print("⚠️ No clear foot plant frame found.")
        return None

    # --- Rewritten Swing Start Detection ---
    def _detect_swing_start_adaptive(
        self,
        data,
        foot_plant_frame: int | None,
        video_length: int,
        handedness: str,
        *,
        fps: float | None = None,
        smooth_sigma: float = 0.6,
        min_dx_units: float = 0.0015,
        min_center_step_units: float = 0.001,
        shoulder_weight: float = 0.7,
        hip_weight: float = 0.3,
        debug: bool = True,
        return_debug: bool = False
    ):
        if foot_plant_frame is None:
            if debug:
                print("❌ No foot plant frame provided.")
            return None

        def _smooth1d(x, sigma):
            x = np.asarray(x, float)
            if sigma and sigma > 0:
                n = len(x)
                idx = np.arange(n)
                m = np.isfinite(x)
                if n and not m.all() and m.any():
                    x = x.copy()
                    x[~m] = np.interp(idx[~m], idx[m], x[m])
                x = gaussian_filter1d(x, sigma=sigma)
            return x

        frames = np.asarray(data["frames"])
        plant_i = int(np.searchsorted(frames, foot_plant_frame, side="left"))
        if plant_i >= len(frames) - 2:
            if debug:
                print("❌ Not enough frames after foot plant.")
            return None

        max_after = int(3 * fps) if (fps and fps > 0) else 90
        search_end_i = min(len(frames) - 1, plant_i + max_after)
        if debug:
           print(f"🔎 Swing-start scan window: idx[{plant_i + 1}..{search_end_i}]")

        Lx = _smooth1d(data["lead_wrist_x"], smooth_sigma)
        Bx = _smooth1d(data["back_wrist_x"], smooth_sigma)
        Hx = 0.5 * (Lx + Bx)

        Lsx = _smooth1d(data["lead_shoulder_x"], 0.8)
        Bsx = _smooth1d(data["back_shoulder_x"], 0.8)
        Shx = 0.5 * (Lsx + Bsx)

        Hcx = _smooth1d(data.get("hip_center_x", [np.nan] * len(Shx)), 1.0)
        use_hip = np.isfinite(Hcx).sum() > 0.7 * len(Hcx)
        CenterX = shoulder_weight * Shx + hip_weight * Hcx if use_hip else Shx

        center_fp = float(CenterX[plant_i])
        sign = +1.0 if handedness == "right" else (-1.0 if handedness == "left" else None)

        debug_info = []

        for i in range(plant_i + 1, search_end_i + 1):
            dx = Hx[i] - Hx[i - 1]
            cx_prev = abs(Hx[i - 1] - center_fp)
            cx_curr = abs(Hx[i] - center_fp)
            toward_center = (cx_curr <= cx_prev - abs(min_center_step_units))
            dir_ok = (sign is None) or (sign * dx >= abs(min_dx_units))
            frame_num = int(frames[i])

            if debug:
                print(
                    f"🧪 Frame {frame_num} (idx {i}): "
                    f"Hx={Hx[i]:.5f}, dx={dx:+.5f}, "
                    f"|Hx-center_fp|: {cx_prev:.5f} → {cx_curr:.5f} "
                    f"{'↓' if toward_center else '↔'}, "
                    f"dir_ok={dir_ok}"
                )

            debug_info.append({
                "frame": frame_num,
                "index": i,
                "hx": float(Hx[i]),
                "dx": float(dx),
                "cx_prev": float(cx_prev),
                "cx_curr": float(cx_curr),
                "toward_center": toward_center,
                "dir_ok": dir_ok,
                "passes": toward_center and dir_ok
            })

            if toward_center and dir_ok:
                if debug:
                    print(f"✅ Swing start at frame {frame_num} (idx {i})")
                return frame_num if not return_debug else (frame_num, debug_info)

        if debug:
            print("🔻 No inward hand step toward FP center found.")
        return None if not return_debug else (None, debug_info)



    def _detect_contact_adaptive(self, data, swing_start, video_length, handedness: str):
        """Contact detection using elbow angle instead of arm length."""

        if swing_start is None:
            print("❌ _detect_contact_adaptive: swing_start is None")
            return None

        # --- locate swing_start index ---
        start_idx = next((i for i, f in enumerate(data['frames']) if f >= swing_start), 0)

        # --- fixed window after swing start ---
        WINDOW_SECONDS = 4.0
        fps_est = 30.0
        window_frames = int(round(fps_est * WINDOW_SECONDS))
        end_idx = min(start_idx + window_frames, len(data['lead_wrist_x']) - 1)

        if end_idx - start_idx < 6:
            print("🚫 contact: too few frames in window after swing_start")
            return None

        # --- smoothing helper ---
        def _smooth1d(x, sigma):
            x = np.asarray(x, float)
            if sigma and sigma > 0:
                n = len(x)
                idx = np.arange(n)
                m = np.isfinite(x)
                if n and (not m.all()) and m.any():
                    x = x.copy()
                    x[~m] = np.interp(idx[~m], idx[m], x[m])
                x = gaussian_filter1d(x, sigma=sigma)
            return x

        # Smooth coordinates
        lsx = _smooth1d(data['lead_shoulder_x'], 0.8)
        lsy = _smooth1d(data['lead_shoulder_y'], 0.8)
        lex = _smooth1d(data['lead_elbow_x'],   0.8)
        ley = _smooth1d(data['lead_elbow_y'],   0.8)
        lwx = _smooth1d(data['lead_wrist_x'],   0.8)
        lwy = _smooth1d(data['lead_wrist_y'],   0.8)

        # === Elbow angle calculation ===
        def elbow_angle_deg(sx, sy, ex, ey, wx, wy):
            SE = np.sqrt((sx - ex)**2 + (sy - ey)**2)
            EW = np.sqrt((ex - wx)**2 + (ey - wy)**2)
            SW = np.sqrt((sx - wx)**2 + (sy - wy)**2)
            if SE == 0 or EW == 0:
                return np.nan
            cos_theta = (SE**2 + EW**2 - SW**2) / (2 * SE * EW)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            return np.degrees(np.arccos(cos_theta))

        elbow_angles = np.array([
            elbow_angle_deg(lsx[i], lsy[i], lex[i], ley[i], lwx[i], lwy[i])
            for i in range(len(lsx))
        ])

        # Threshold for "almost straight" arm
        ANGLE_THRESH = 160.0
        SECONDARY_ANGLE_THRESH = 145.0


        # Hands-ahead-of-face check
        hx = 0.5 * (np.asarray(data['lead_wrist_x'], float) + np.asarray(data['back_wrist_x'], float))
        hx_s = _smooth1d(hx, 0.6)
        face_x = np.asarray(data.get('face_center_x', []), float)
        have_face = (len(face_x) == len(hx_s))
        face_x_s = _smooth1d(face_x, 0.8) if have_face else face_x
        sign = +1.0 if handedness == 'right' else (-1.0 if handedness == 'left' else None)
        margin = 0.065

        def hands_ahead(idx: int) -> bool:
            if not (have_face and sign is not None):
                return False
            if not (np.isfinite(hx_s[idx]) and np.isfinite(face_x_s[idx])):
                return False
            diff = sign * (hx_s[idx] - face_x_s[idx])
            return diff >= margin

        bex = _smooth1d(data['back_elbow_x'], 0.6)

        def back_elbow_ahead(idx: int) -> bool:
            if not (have_face and sign is not None):
                return False
            if not (np.isfinite(bex[idx]) and np.isfinite(face_x_s[idx])):
                return False
            diff = sign * (bex[idx] - face_x_s[idx])
            return diff >= margin


        # --- DEBUG print ---
        """
        print("\n🛠 DEBUG: Frame-by-frame contact search window")
        print(" idx | frame | elbow_angle | near_full? | hands_ahead? | elbow_ahead? | elbow_ahead_dist")
        print("-----|-------|-------------|------------|--------------|--------------|------------------")
        for idx in range(start_idx, end_idx + 1):
            angle = elbow_angles[idx]
            nf = np.isfinite(angle) and angle >= ANGLE_THRESH
            ahead = hands_ahead(idx)

            # Compute elbow diff
            if have_face and sign is not None and np.isfinite(bex[idx]) and np.isfinite(face_x_s[idx]):
                elbow_diff = sign * (bex[idx] - face_x_s[idx])
                elbow_ahead = elbow_diff >= margin
            else:
                elbow_diff = float('nan')
                elbow_ahead = False

            print(f"{idx:4d} | {int(data['frames'][idx]):5d} | {angle:11.2f} | {str(nf):>10} | {str(ahead):>12} | {str(elbow_ahead):>12} | {elbow_diff:16.4f}")
        """

        

        # === Selection logic ===
        # Step 1: earliest frame with straight arm & hands ahead
        earliest_idx = None
        for idx in range(start_idx, end_idx + 1):
            if np.isfinite(elbow_angles[idx]):
                angle = elbow_angles[idx]
                if (angle >= ANGLE_THRESH and hands_ahead(idx)) or \
                (angle >= SECONDARY_ANGLE_THRESH and back_elbow_ahead(idx)):
                    earliest_idx = idx
                    break


        if earliest_idx is not None:
            print(f"🎯 contact: earliest straight-arm & ahead-of-face at idx={earliest_idx} "
                f"(frame={int(data['frames'][earliest_idx])}, angle={elbow_angles[earliest_idx]:.2f}°)")
            return int(data['frames'][earliest_idx])

        # Step 2: fallback to straightest arm
        valid_idxs = [i for i in range(start_idx, end_idx + 1) if np.isfinite(elbow_angles[i])]
        if valid_idxs:
            k = max(valid_idxs, key=lambda i: elbow_angles[i])
            print(f"ℹ️ no ahead-of-face match; using straightest arm idx={k} "
                f"(frame={int(data['frames'][k])}, angle={elbow_angles[k]:.2f}°)")
            return int(data['frames'][k])

        print("🚫 No valid elbow angles for contact detection")
        return None

    
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


def calculate_attack_angle_from_landmarks(landmarks, handedness: str) -> float | None:
    """
    Estimate attack angle proxy using shoulder tilt.
    
    Returns a proxy value that mimics attack angle ranges:
    - Positive values (4-16°): Upward swing (good for line drives)
    - Zero (0°): Level swing
    - Negative values (-5° to -15°): Downward swing (chopping)
    
    Target range for line drives: 6-14° (similar to actual attack angle)
    
    Args:
        landmarks: MediaPipe pose landmarks
        handedness: "right" or "left" 
        
    Returns:
        Attack angle proxy in degrees, or None if calculation fails
    """
    try:
        if handedness == "right":
            front_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            rear_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        elif handedness == "left":
            front_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            rear_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        else:
            return None

        # Simple logic: compare Y coordinates
        # In CV coordinates: higher Y = lower in image
        # If rear shoulder Y > front shoulder Y → rear shoulder is lower → upward swing
        dy = rear_shoulder.y - front_shoulder.y
        dx = rear_shoulder.x - front_shoulder.x
        
        print(f"DEBUG: dy={dy:.3f} (positive = rear shoulder lower)")
        print(f"DEBUG: dx={dx:.3f}")
        
        # Calculate angle using slope
        if abs(dx) < 0.001:  # Avoid division by zero
            angle_proxy = 0
        else:
            slope = dy / dx
            angle_deg = np.degrees(np.arctan(slope))
            print(f"DEBUG: slope={slope:.3f}, angle_deg={angle_deg:.1f}")
            
            # The sign of the angle depends on both dy and dx
            # We want: rear shoulder lower (dy > 0) = upward swing = positive proxy
            if dy > 0:  # Rear shoulder lower = upward swing
                angle_proxy = abs(angle_deg) * 0.2  # Always positive
            else:  # Rear shoulder higher = downward swing  
                angle_proxy = -abs(angle_deg) * 0.2  # Always negative
        
        print(f"DEBUG: Final attack angle proxy: {angle_proxy:.1f}°")
        
        return round(angle_proxy, 1)
        
    except Exception as e:
        print(f"⚠️ Error calculating attack angle proxy: {e}")
        return None


# ---------------- Rotation fix ----------------

def fix_rotation_if_needed(input_path: str) -> tuple[str, bool, int]:
    """
    Uses ffprobe to read rotation metadata and, if needed, writes an upright copy
    to a new temp file and returns (path, was_rotated, rotation_deg).
    If no rotation needed, returns (input_path, False, 0).
    """
    # 1) Probe rotation metadata
    try:
        probe = ffmpeg.probe(input_path)
    except Exception as e:
        print(f"⚠️ ffprobe failed, skipping rotation fix: {e}")
        return input_path, False, 0

    vstreams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    if not vstreams:
        print("⚠️ No video streams found by ffprobe; skipping rotation fix.")
        return input_path, False, 0
    st = vstreams[0]

    rotation = 0
    src = "none"
    rot_tag = (st.get("tags") or {}).get("rotate")
    if rot_tag is not None:
        try:
            rotation = int(rot_tag) % 360
            src = "tags.rotate"
        except:
            pass
    if rotation == 0:
        for sd in (st.get("side_data_list") or []):
            if sd.get("side_data_type") == "Display Matrix" and sd.get("rotation") is not None:
                try:
                    rotation = int(sd["rotation"]) % 360
                    src = "side_data_list.Display Matrix"
                    break
                except:
                    pass

    print(f"🔎 ffprobe rotation: {rotation}° (source: {src})")

    if rotation == 0:
        print("📐 No rotation needed.")
        return input_path, False, 0

    # 2) Physically rotate frames with OpenCV (no metadata involved)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("⚠️ OpenCV failed to open input; skipping rotation.")
        return input_path, False, rotation

    in_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    in_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # FIXED: Correct output dimensions based on rotation
    if rotation in (90, 270):
        out_size = (in_h, in_w)  # swap dimensions for 90/270 degree rotations
    else:
        out_size = (in_w, in_h)  # keep same for 180 degree

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    out = cv2.VideoWriter(out_path, fourcc, fps, out_size)

    # FIXED: Correct rotation mapping
    # Metadata rotation describes how the video SHOULD be rotated to appear upright
    # We need to apply the OPPOSITE rotation to make it upright
    if rotation == 90:
        # Video was recorded rotated 90° CW, so rotate 90° CCW to fix it
        rot_code = cv2.ROTATE_90_COUNTERCLOCKWISE
    elif rotation == 270:
        # Video was recorded rotated 90° CCW, so rotate 90° CW to fix it  
        rot_code = cv2.ROTATE_90_CLOCKWISE
    elif rotation == 180:
        rot_code = None  # handled separately with flip
    else:
        print(f"⚠️ Unsupported rotation {rotation}°; using original.")
        cap.release()
        out.release()
        try:
            os.remove(out_path)
        except:
            pass
        return input_path, False, rotation

    frames = 0
    print(f"🔄 Applying rotation correction: {rotation}° metadata -> {rot_code if rot_code else 'flip'}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply the correction rotation
        if rotation == 180:
            frame = cv2.flip(frame, -1)  # 180° rotate via flip
        else:
            frame = cv2.rotate(frame, rot_code)
            
        out.write(frame)
        frames += 1

    cap.release()
    out.release()

    if frames == 0:
        # If something went wrong, fall back to original
        try:
            os.remove(out_path)
        except:
            pass
        print("⚠️ No frames written during rotation; using original.")
        return input_path, False, rotation

    print(f"✅ Wrote upright copy (pixel-rotated): {out_path} ({frames} frames)")
    return out_path, True, rotation


# ---------------- Analyzer entry (uploads rotated copy if present) ----------------

async def analyze_video_from_url(url: str):
    print(f"🔗 Downloading video from URL: {url}")
    feedback = []
    biomarker_results = {}

    # 1) Download original
    original_path = await _download_temp(url)

    # 2) Normalize rotation (always gives us an upright copy)
    fixed_path, was_rotated, rotation_deg = fix_rotation_if_needed(original_path)

    # 3) Analyze *only the local upright copy*
    cap = cv2.VideoCapture(fixed_path)
    if not cap.isOpened():
        return [{"message": "Failed to read video file"}], {}

    analyzer = RobustBaseballSwingAnalyzer()

    landmarks_over_time = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (total_frames / fps) if fps > 0 else 0.0
    print(f"📹 Processing from local upright copy: {fixed_path}")
    print(f"📹 {total_frames} frames at {fps:.1f} FPS ({duration:.1f}s)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = analyzer.pose.process(image_rgb)
        if results.pose_landmarks:
            landmarks_over_time.append(results.pose_landmarks.landmark)

    cap.release()

    if len(landmarks_over_time) < 10:
        return ([{"message": "Insufficient pose detection. Ensure clear view of full body throughout the swing."}], {})

    print(f"🧠 Analyzing {len(landmarks_over_time)} frames with pose data...")

    # ----- your existing analysis below -----
    handedness_result = analyzer.detect_handedness_fusion(landmarks_over_time)
    handedness = handedness_result["handedness"]
    print(f"🤚 Detected handedness: {handedness} (confidence: {handedness_result.get('confidence', 0):.2f})")

    swing_phases = analyzer.detect_precise_swing_timing(landmarks_over_time, handedness)

    foot_plant_frame = swing_phases.get("foot_plant")

    print("⏱️ Swing phases detected:")
    for phase, frame in swing_phases.items():
        if frame is not None:
            t = frame / fps if fps and fps > 0 else frame * 0.033
            print(f"   • {phase.replace('_',' ').title()}: Frame {frame} ({t:.2f}s)")
        else:
            print(f"   • {phase.replace('_',' ').title()}: Not detected")

    
    detected_phases = [p for p, f in swing_phases.items() if f is not None]
    if len(detected_phases) < 3:
        feedback.append({
            "frame": "overall",
            "issue": f"Only {len(detected_phases)} of 5 swing phases detected clearly",
            "suggested_drill": "Ensure video captures complete swing from stance through follow-through",
            "severity": "medium"
        })

    if swing_phases.get("swing_start") and swing_phases.get("contact"):
        swing_duration = swing_phases["contact"] - swing_phases["swing_start"]
        if swing_duration < 8:
            feedback.append({
                "frame": swing_phases["swing_start"],
                "issue": f"Very quick swing ({swing_duration} frames)",
                "suggested_drill": "Consider working on tempo and timing control",
                "severity": "low"
            })

    if handedness == "unknown":
        feedback.append({
            "frame": "setup",
            "issue": "Could not determine handedness reliably",
            "suggested_drill": "Ensure clear side view showing batting stance and initial setup",
            "severity": "high"
        })
    elif handedness_result.get("confidence", 0) < 0.7:
        feedback.append({
            "frame": "setup",
            "issue": f"Handedness detection has moderate confidence ({handedness_result['confidence']:.2f})",
            "suggested_drill": "Verify analysis results - ensure clear side view of batting stance",
            "severity": "low"
        })

    #data_quality = separation_analysis.get('data_quality', 0.0)
  

    biomarker_results = {
        "handedness": handedness_result,
        "swing_phases": swing_phases,
        "video_info": {
            "fps": fps,
            "frames_analyzed": len(landmarks_over_time),
            "duration_seconds": duration
        },
        "media": {
            "input_url": url,
            "rotation_deg": rotation_deg,
            "was_rotated": was_rotated
        }
    }

    print("✅ Analysis complete (rotation handled locally).")
    return feedback or [{"message": "No major issues detected."}], biomarker_results, fixed_path

# ---------------- Annotated video generator (works with upright video) ----------------

async def generate_annotated_video(input_path: str, swing_phases: dict, handedness) -> str:
    """
    Generate an annotated upright video saved as 'annotated_output.mp4'.
    Handles rotation correction if needed.
    """

    # Ensure the video is upright
    fixed_path, was_rotated, rotation_deg = fix_rotation_if_needed(input_path)

    cap = cv2.VideoCapture(fixed_path)
    if not cap.isOpened():
        raise Exception("Failed to open video for annotation")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    print(f"📐 Input video (upright): {width}x{height} at {fps:.1f} FPS ({total_frames} frames)")

    pose = mp_pose.Pose(static_image_mode=False)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = "annotated_output.mp4"
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    frames_written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Annotate hip-shoulder separation at foot plant
            if swing_phases.get("foot_plant") == frame_idx:
                landmarks = results.pose_landmarks.landmark
                def denorm(p): return (int(p.x * width), int(p.y * height))

                ls = denorm(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
                rs = denorm(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
                lh = denorm(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
                rh = denorm(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])

                cv2.line(frame, ls, rs, (0, 255, 255), 3)   # shoulders: yellow
                cv2.line(frame, lh, rh, (255, 0, 255), 3)   # hips: magenta

                shoulder_angle = np.degrees(np.arctan2(rs[1] - ls[1], rs[0] - ls[0]))
                hip_angle = np.degrees(np.arctan2(rh[1] - lh[1], rh[0] - lh[0]))
                separation = abs(((shoulder_angle - hip_angle + 180) % 360) - 180)

                cv2.putText(
                    frame, f"Hip-Shoulder Sep: {separation:.1f} degrees",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2
                )

        if swing_phases.get("contact") == frame_idx:
                # Calculate attack angle
                attack_angle = calculate_attack_angle_from_landmarks(results.pose_landmarks.landmark, handedness)
                
                if attack_angle is not None:
                    cv2.putText(
                        frame,
                        f"Attack Angle: {attack_angle:.1f} degrees",
                        (30, 130),  # Y-position below other annotations
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 0),
                        2
                    )


        # Add swing phase label if present
        for phase, f in swing_phases.items():
            if f == frame_idx:
                label = phase.replace("_", " ").title()
                color = {
                    "stride_start": (0, 0, 0),
                    "foot_plant": (0, 0, 0),
                    "swing_start": (0, 0, 0),
                    "contact": (0, 0, 0),
                    "follow_through": (0, 0, 0)
                }.get(phase, (0, 0, 0))
                cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1)

        # Frame counter
        cv2.putText(frame, f"Frame {frame_idx}", (30, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        writer.write(frame)
        frames_written += 1
        frame_idx += 1

    cap.release()
    writer.release()
    pose.close()

    print(f"✅ Annotated video written to: {output_path} ({frames_written} frames)")

    # Clean up rotated temp file if needed
    try:
        if was_rotated and os.path.exists(fixed_path):
            os.remove(fixed_path)
            print(f"🗑️ Cleaned up rotated temp file: {fixed_path}")
    except Exception as e:
        print(f"⚠️ Cleanup error: {e}")

    return output_path

# ---------------- Download helper ----------------

async def _download_temp(url: str) -> str:
    """
    Download the video to a temp .mp4 and return the local path.
    No rotation detection/correction. Generous timeout for mobile uploads.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_path = tmp.name
    tmp.close()

    timeout = httpx.Timeout(120.0, connect=20.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                async for chunk in resp.aiter_bytes():
                    f.write(chunk)
    
    print(f"📥 Downloaded video to: {tmp_path}")
    return tmp_path