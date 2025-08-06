import cv2
import numpy as np
import mediapipe as mp
import tempfile
import httpx
import os
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import json
from typing import Dict, List, Tuple, Optional

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

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

    
    def _analyze_batting_stance_comprehensive(self, stance_frames):
        """Comprehensive batting stance analysis using multiple body segments."""
        left_indicators = []
        right_indicators = []
        
        for lm in stance_frames:
            try:
                # Get normalized positions
                landmarks_normalized = {}
                landmarks_to_check = [
                    'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP',
                    'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
                    'LEFT_WRIST', 'RIGHT_WRIST'
                ]
                
                valid_landmarks = True
                for landmark_name in landmarks_to_check:
                    landmark_idx = getattr(mp_pose.PoseLandmark, landmark_name).value
                    normalized = self.normalize_landmark_position(lm[landmark_idx], lm)
                    if normalized is None:
                        valid_landmarks = False
                        break
                    landmarks_normalized[landmark_name] = normalized
                
                if not valid_landmarks:
                    continue
                
                # Multi-segment depth consistency analysis
                segments = {
                    'upper': (landmarks_normalized['LEFT_SHOULDER']['x'] - landmarks_normalized['RIGHT_SHOULDER']['x']),
                    'core': (landmarks_normalized['LEFT_HIP']['x'] - landmarks_normalized['RIGHT_HIP']['x']),
                    'lower': (landmarks_normalized['LEFT_KNEE']['x'] - landmarks_normalized['RIGHT_KNEE']['x']),
                    'feet': (landmarks_normalized['LEFT_ANKLE']['x'] - landmarks_normalized['RIGHT_ANKLE']['x']),
                    'hands': (landmarks_normalized['LEFT_WRIST']['x'] - landmarks_normalized['RIGHT_WRIST']['x'])
                }
                
                # Check consistency across body segments
                segment_values = list(segments.values())
                consistency = 1.0 - (np.std(segment_values) / (np.mean(np.abs(segment_values)) + 0.1))
                avg_depth = np.mean(segment_values)
                
                # Additional stance width analysis for validation
                stance_width = abs(landmarks_normalized['LEFT_ANKLE']['y'] - landmarks_normalized['RIGHT_ANKLE']['y'])
                
                # Strong indicator: consistent depth + appropriate stance width
                if consistency > 0.7 and abs(avg_depth) > 0.15 and stance_width > 0.2:
                    confidence = consistency * min(abs(avg_depth) * 3, 1.0) * min(stance_width, 1.0)
                    
                    if avg_depth > 0:  # Left side back = right handed
                        right_indicators.append(confidence)
                    else:  # Right side back = left handed
                        left_indicators.append(confidence)
                        
            except:
                continue
        
        total_frames = len(left_indicators) + len(right_indicators)
        if total_frames < max(3, len(stance_frames) // 4):
            return {"result": "unknown", "confidence": 0.0, "debug": "Insufficient consistent stance frames"}
        
        left_score = np.mean(left_indicators) if left_indicators else 0
        right_score = np.mean(right_indicators) if right_indicators else 0
        
        # Confidence based on consistency and strength of evidence
        frame_consistency = total_frames / len(stance_frames)
        confidence = max(left_score, right_score) * frame_consistency
        
        if left_score > right_score and left_score > 0.4:
            return {"result": "left", "confidence": min(confidence, 0.95), 
                   "debug": f"Left stance score: {left_score:.3f}, frames: {len(left_indicators)}/{len(stance_frames)}"}
        elif right_score > left_score and right_score > 0.4:
            return {"result": "right", "confidence": min(confidence, 0.95),
                   "debug": f"Right stance score: {right_score:.3f}, frames: {len(right_indicators)}/{len(stance_frames)}"}
        else:
            return {"result": "unknown", "confidence": 0.0, 
                   "debug": f"Inconclusive scores - Left: {left_score:.3f}, Right: {right_score:.3f}"}
    
    def _analyze_body_asymmetry(self, stance_frames):
        """Analyze natural body asymmetry in batting stance."""
        asymmetry_scores = {"left_handed": [], "right_handed": []}
        
        for lm in stance_frames:
            try:
                # Key asymmetry indicators
                left_shoulder_norm = self.normalize_landmark_position(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value], lm)
                right_shoulder_norm = self.normalize_landmark_position(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], lm)
                left_hip_norm = self.normalize_landmark_position(lm[mp_pose.PoseLandmark.LEFT_HIP.value], lm)
                right_hip_norm = self.normalize_landmark_position(lm[mp_pose.PoseLandmark.RIGHT_HIP.value], lm)
                left_knee_norm = self.normalize_landmark_position(lm[mp_pose.PoseLandmark.LEFT_KNEE.value], lm)
                right_knee_norm = self.normalize_landmark_position(lm[mp_pose.PoseLandmark.RIGHT_KNEE.value], lm)
                
                if not all([left_shoulder_norm, right_shoulder_norm, left_hip_norm, right_hip_norm, left_knee_norm, right_knee_norm]):
                    continue
                
                # Analyze weight distribution asymmetry
                left_weight_line = np.mean([left_shoulder_norm['x'], left_hip_norm['x'], left_knee_norm['x']])
                right_weight_line = np.mean([right_shoulder_norm['x'], right_hip_norm['x'], right_knee_norm['x']])
                
                # Shoulder height asymmetry (natural in batting stance)
                shoulder_height_diff = left_shoulder_norm['y'] - right_shoulder_norm['y']
                
                # Hip angle
                hip_angle = np.arctan2(right_hip_norm['y'] - left_hip_norm['y'], 
                                     right_hip_norm['x'] - left_hip_norm['x'])
                
                # Combined asymmetry score
                weight_asymmetry = left_weight_line - right_weight_line
                
                # Right-handed: left side typically back, left shoulder often lower
                if weight_asymmetry > 0.1 and shoulder_height_diff < -0.05:
                    asymmetry_scores["right_handed"].append(abs(weight_asymmetry) + abs(shoulder_height_diff))
                # Left-handed: right side back, right shoulder often lower  
                elif weight_asymmetry < -0.1 and shoulder_height_diff > 0.05:
                    asymmetry_scores["left_handed"].append(abs(weight_asymmetry) + abs(shoulder_height_diff))
                    
            except:
                continue
        
        left_score = np.mean(asymmetry_scores["left_handed"]) if asymmetry_scores["left_handed"] else 0
        right_score = np.mean(asymmetry_scores["right_handed"]) if asymmetry_scores["right_handed"] else 0
        
        total_evidence = len(asymmetry_scores["left_handed"]) + len(asymmetry_scores["right_handed"])
        
        if total_evidence < 3:
            return {"result": "unknown", "confidence": 0.0, "debug": "Insufficient asymmetry data"}
        
        confidence = max(left_score, right_score) * (total_evidence / len(stance_frames))
        
        if left_score > right_score and left_score > 0.15:
            return {"result": "left", "confidence": min(confidence, 0.8), 
                   "debug": f"Left asymmetry: {left_score:.3f}, evidence frames: {len(asymmetry_scores['left_handed'])}"}
        elif right_score > left_score and right_score > 0.15:
            return {"result": "right", "confidence": min(confidence, 0.8),
                   "debug": f"Right asymmetry: {right_score:.3f}, evidence frames: {len(asymmetry_scores['right_handed'])}"}
        else:
            return {"result": "unknown", "confidence": 0.0, "debug": "Insufficient asymmetry difference"}
    
    def _analyze_limb_positioning(self, stance_frames):
        """Analyze limb positioning patterns specific to batting stance."""
        positioning_votes = {"left": 0, "right": 0}
        total_analyses = 0
        
        for lm in stance_frames:
            try:
                # Foot positioning analysis
                left_ankle_norm = self.normalize_landmark_position(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value], lm)
                right_ankle_norm = self.normalize_landmark_position(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value], lm)
                left_knee_norm = self.normalize_landmark_position(lm[mp_pose.PoseLandmark.LEFT_KNEE.value], lm)
                right_knee_norm = self.normalize_landmark_position(lm[mp_pose.PoseLandmark.RIGHT_KNEE.value], lm)
                
                if not all([left_ankle_norm, right_ankle_norm, left_knee_norm, right_knee_norm]):
                    continue
                
                # Lead foot typically forward and weight-bearing leg more vertical
                foot_forward_diff = left_ankle_norm['x'] - right_ankle_norm['x']
                knee_vertical_diff = abs(left_knee_norm['x'] - left_ankle_norm['x']) - abs(right_knee_norm['x'] - right_ankle_norm['x'])
                
                # Strong foot positioning signal
                if abs(foot_forward_diff) > 0.2:
                    if foot_forward_diff > 0:  # Left foot forward = right handed
                        positioning_votes["right"] += 2
                    else:  # Right foot forward = left handed
                        positioning_votes["left"] += 2
                
                # Knee alignment signal (weight-bearing leg more vertical)
                if abs(knee_vertical_diff) > 0.15:
                    if knee_vertical_diff < 0:  # Left leg more vertical = left lead = right handed
                        positioning_votes["right"] += 1
                    else:  # Right leg more vertical = right lead = left handed
                        positioning_votes["left"] += 1
                
                total_analyses += 1
                
            except:
                continue
        
        if total_analyses < 3:
            return {"result": "unknown", "confidence": 0.0, "debug": "Insufficient limb positioning data"}
        
        total_votes = positioning_votes["left"] + positioning_votes["right"]
        
        if positioning_votes["left"] > positioning_votes["right"]:
            confidence = positioning_votes["left"] / (total_analyses * 3)  # Max 3 points per frame
            return {"result": "left", "confidence": min(confidence, 0.85), 
                   "debug": f"Left positioning votes: {positioning_votes['left']}/{total_votes}"}
        elif positioning_votes["right"] > positioning_votes["left"]:
            confidence = positioning_votes["right"] / (total_analyses * 3)
            return {"result": "right", "confidence": min(confidence, 0.85),
                   "debug": f"Right positioning votes: {positioning_votes['right']}/{total_votes}"}
        else:
            return {"result": "unknown", "confidence": 0.0, "debug": f"Tied positioning votes: {positioning_votes}"}
    
    def _analyze_movement_initiation(self, landmarks_over_time):
        """Analyze initial movement patterns to identify swing initiation side."""
        video_length = len(landmarks_over_time)
        
        # Adaptive frame selection based on video length
        if video_length <= 40:
            baseline_end = max(8, video_length // 4)
            movement_start = baseline_end
            movement_end = min(video_length - 5, baseline_end + 15)
        else:
            baseline_end = 15
            movement_start = 15
            movement_end = 35
        
        if movement_end <= movement_start:
            return {"result": "unknown", "confidence": 0.0, "debug": "Insufficient frames for movement analysis"}
        
        baseline_frames = landmarks_over_time[:baseline_end]
        movement_frames = landmarks_over_time[movement_start:movement_end]
        
        # Calculate baseline positions for key points
        baseline_positions = {}
        landmarks_to_track = ['LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_SHOULDER', 'RIGHT_SHOULDER']
        
        for landmark_name in landmarks_to_track:
            positions = []
            for lm in baseline_frames:
                try:
                    landmark_idx = getattr(mp_pose.PoseLandmark, landmark_name).value
                    normalized = self.normalize_landmark_position(lm[landmark_idx], lm)
                    if normalized:
                        positions.append((normalized['x'], normalized['y']))
                except:
                    continue
            
            if len(positions) >= 3:
                baseline_positions[landmark_name] = {
                    'x': np.mean([p[0] for p in positions]),
                    'y': np.mean([p[1] for p in positions]),
                    'std_x': np.std([p[0] for p in positions]),
                    'std_y': np.std([p[1] for p in positions])
                }
        
        if len(baseline_positions) < 4:
            return {"result": "unknown", "confidence": 0.0, "debug": "Insufficient baseline data"}
        
        # Analyze movement from baseline
        movement_scores = {"left_initiation": [], "right_initiation": []}
        
        for lm in movement_frames:
            try:
                frame_movements = {}
                for landmark_name in landmarks_to_track:
                    if landmark_name not in baseline_positions:
                        continue
                    
                    landmark_idx = getattr(mp_pose.PoseLandmark, landmark_name).value
                    normalized = self.normalize_landmark_position(lm[landmark_idx], lm)
                    if normalized:
                        baseline = baseline_positions[landmark_name]
                        movement_magnitude = np.sqrt(
                            (normalized['x'] - baseline['x'])**2 + 
                            (normalized['y'] - baseline['y'])**2
                        )
                        # Normalize by baseline variability
                        movement_significance = movement_magnitude / (baseline['std_x'] + baseline['std_y'] + 0.05)
                        frame_movements[landmark_name] = movement_significance
                
                if len(frame_movements) >= 4:
                    # Analyze which side shows more movement initiation
                    left_side_movement = np.mean([frame_movements.get(f'LEFT_{part}', 0) for part in ['ANKLE', 'HIP', 'SHOULDER']])
                    right_side_movement = np.mean([frame_movements.get(f'RIGHT_{part}', 0) for part in ['ANKLE', 'HIP', 'SHOULDER']])
                    
                    movement_diff = abs(left_side_movement - right_side_movement)
                    
                    if movement_diff > 0.3:  # Significant asymmetric movement
                        if left_side_movement > right_side_movement:
                            # Left side moves first = lead leg = right handed
                            movement_scores["right_initiation"].append(movement_diff)
                        else:
                            # Right side moves first = lead leg = left handed
                            movement_scores["left_initiation"].append(movement_diff)
                
            except:
                continue
        
        left_score = np.mean(movement_scores["left_initiation"]) if movement_scores["left_initiation"] else 0
        right_score = np.mean(movement_scores["right_initiation"]) if movement_scores["right_initiation"] else 0
        
        total_evidence = len(movement_scores["left_initiation"]) + len(movement_scores["right_initiation"])
        
        if total_evidence < 3:
            return {"result": "unknown", "confidence": 0.0, "debug": "Insufficient movement evidence"}
        
        confidence = max(left_score, right_score) * min(total_evidence / len(movement_frames), 1.0)
        
        if left_score > right_score and left_score > 0.4:
            return {"result": "left", "confidence": min(confidence, 0.8),
                   "debug": f"Left initiation score: {left_score:.3f}, evidence: {len(movement_scores['left_initiation'])}"}
        elif right_score > left_score and right_score > 0.4:
            return {"result": "right", "confidence": min(confidence, 0.8),
                   "debug": f"Right initiation score: {right_score:.3f}, evidence: {len(movement_scores['right_initiation'])}"}
        else:
            return {"result": "unknown", "confidence": 0.0, "debug": "Insufficient movement asymmetry"}
    
    def _analyze_depth_relationships(self, stance_frames):
        """Analyze depth relationships between body parts for perspective-aware handedness detection."""
        depth_consistency_scores = {"left": [], "right": []}
        
        for lm in stance_frames:
            try:
                # Get key landmarks
                landmarks_normalized = {}
                key_landmarks = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 
                               'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP']
                
                valid_frame = True
                for landmark_name in key_landmarks:
                    landmark_idx = getattr(mp_pose.PoseLandmark, landmark_name).value
                    normalized = self.normalize_landmark_position(lm[landmark_idx], lm)
                    if normalized is None:
                        valid_frame = False
                        break
                    landmarks_normalized[landmark_name] = normalized
                
                if not valid_frame:
                    continue
                
                # Analyze depth relationships
                # In side view, one side of body should consistently be behind the other
                left_chain_x = [landmarks_normalized[f'LEFT_{part}']['x'] for part in ['SHOULDER', 'ELBOW', 'WRIST', 'HIP']]
                right_chain_x = [landmarks_normalized[f'RIGHT_{part}']['x'] for part in ['SHOULDER', 'ELBOW', 'WRIST', 'HIP']]
                
                left_avg_depth = np.mean(left_chain_x)
                right_avg_depth = np.mean(right_chain_x)
                depth_difference = left_avg_depth - right_avg_depth
                
                # Check consistency within each side
                left_consistency = 1.0 - (np.std(left_chain_x) / 0.5)  # Lower std = more consistent
                right_consistency = 1.0 - (np.std(right_chain_x) / 0.5)
                overall_consistency = (left_consistency + right_consistency) / 2
                
                # Strong depth separation + good consistency = reliable indicator
                if abs(depth_difference) > 0.25 and overall_consistency > 0.6:
                    confidence_score = abs(depth_difference) * overall_consistency
                    
                    if depth_difference > 0:  # Left side back = right handed
                        depth_consistency_scores["right"].append(confidence_score)
                    else:  # Right side back = left handed
                        depth_consistency_scores["left"].append(confidence_score)
                
            except:
                continue
        
        left_score = np.mean(depth_consistency_scores["left"]) if depth_consistency_scores["left"] else 0
        right_score = np.mean(depth_consistency_scores["right"]) if depth_consistency_scores["right"] else 0
        
        total_evidence = len(depth_consistency_scores["left"]) + len(depth_consistency_scores["right"])
        
        if total_evidence < max(2, len(stance_frames) // 5):
            return {"result": "unknown", "confidence": 0.0, "debug": "Insufficient depth relationship data"}
        
        evidence_ratio = total_evidence / len(stance_frames)
        confidence = max(left_score, right_score) * evidence_ratio
        
        if left_score > right_score and left_score > 0.3:
            return {"result": "left", "confidence": min(confidence, 0.9),
                   "debug": f"Left depth score: {left_score:.3f}, evidence: {len(depth_consistency_scores['left'])}/{len(stance_frames)}"}
        elif right_score > left_score and right_score > 0.3:
            return {"result": "right", "confidence": min(confidence, 0.9),
                   "debug": f"Right depth score: {right_score:.3f}, evidence: {len(depth_consistency_scores['right'])}/{len(stance_frames)}"}
        else:
            return {"result": "unknown", "confidence": 0.0, "debug": f"Insufficient depth separation - Left: {left_score:.3f}, Right: {right_score:.3f}"}
    
    def _combine_handedness_methods_adaptive(self, methods, video_length):
        """Adaptively combine handedness detection methods based on video characteristics."""
        # Adaptive weights based on video length and method reliability
        base_weights = {
            'stance_analysis': 0.30,
            'asymmetry_analysis': 0.25,
            'limb_positioning': 0.20,
            'depth_analysis': 0.15,
            'movement_pattern': 0.10
        }
        
        # Adjust weights based on video length
        if video_length < 30:
            # Shorter videos: rely more on static analysis
            base_weights['stance_analysis'] *= 1.2
            base_weights['asymmetry_analysis'] *= 1.1
            base_weights['movement_pattern'] *= 0.7
        elif video_length > 80:
            # Longer videos: movement analysis becomes more reliable
            base_weights['movement_pattern'] *= 1.3
            base_weights['stance_analysis'] *= 0.9
        
        left_score = 0.0
        right_score = 0.0
        total_weight = 0.0
        debug_info = {}
        
        # Confidence threshold for method inclusion
        min_confidence = max(0.15, 0.3 - (video_length / 200))  # Lower threshold for longer videos
        
        for method_name, method_result in methods.items():
            method_confidence = method_result.get("confidence", 0.0)
            
            if method_confidence > min_confidence:
                base_weight = base_weights.get(method_name, 0.1)
                # Weight by confidence and consistency
                effective_weight = base_weight * method_confidence
                
                debug_info[method_name] = {
                    "result": method_result["result"],
                    "confidence": method_confidence,
                    "weight": effective_weight,
                    "debug": method_result.get("debug", "")
                }
                
                if method_result["result"] == "left":
                    left_score += effective_weight
                elif method_result["result"] == "right":
                    right_score += effective_weight
                
                total_weight += effective_weight
        
        # Adaptive decision thresholds
        min_total_weight = 0.25 if video_length > 50 else 0.35
        decision_threshold_ratio = 0.55 if video_length > 40 else 0.65
        
        if total_weight > min_total_weight:
            decision_threshold = total_weight * decision_threshold_ratio
            
            if left_score > right_score and left_score > decision_threshold:
                final_confidence = min((left_score / total_weight) * 1.1, 0.98)
                return {
                    "handedness": "left",
                    "confidence": final_confidence,
                    "debug": debug_info,
                    "scores": {"left": left_score, "right": right_score, "total_weight": total_weight},
                    "video_length": video_length
                }
            elif right_score > left_score and right_score > decision_threshold:
                final_confidence = min((right_score / total_weight) * 1.1, 0.98)
                return {
                    "handedness": "right",
                    "confidence": final_confidence,
                    "debug": debug_info,
                    "scores": {"left": left_score, "right": right_score, "total_weight": total_weight},
                    "video_length": video_length
                }
        
        return {
            "handedness": "unknown",
            "confidence": 0.0,
            "debug": debug_info,
            "scores": {"left": left_score, "right": right_score, "total_weight": total_weight},
            "video_length": video_length
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
        """Adaptive stride start detection that handles different video lengths and zoom levels."""
        if len(data['lead_ankle_x']) < 8:
            return None
        
        # Adaptive baseline calculation based on video length
        if video_length <= 30:
            baseline_length = max(4, len(data['lead_ankle_x']) // 4)
        elif video_length <= 60:
            baseline_length = max(6, len(data['lead_ankle_x']) // 5)
        else:
            baseline_length = max(8, len(data['lead_ankle_x']) // 6)
        
        # Calculate robust baseline using multiple metrics
        baseline_ankle_x = np.median(data['lead_ankle_x'][:baseline_length])
        baseline_ankle_y = np.median(data['lead_ankle_y'][:baseline_length])
        baseline_knee_x = np.median(data['lead_knee_x'][:baseline_length])
        
        # Adaptive thresholds based on video characteristics and body scale variation
        scale_variation = np.std(data['body_scales'][:baseline_length]) if len(data['body_scales']) > baseline_length else 0.02
        ankle_x_variation = np.std(data['lead_ankle_x'][:baseline_length])
        ankle_y_variation = np.std(data['lead_ankle_y'][:baseline_length])
        
        # Adaptive thresholds that account for natural variation and zoom changes
        threshold_x = max(0.08, ankle_x_variation * 3, scale_variation * 2)
        threshold_y = max(0.06, ankle_y_variation * 2.5, scale_variation * 1.5)
        
        # Multi-point smoothing for noise reduction
        smoothing_sigma = max(1.0, video_length / 80)  # More smoothing for longer videos
        smoothed_ankle_x = gaussian_filter1d(data['lead_ankle_x'], sigma=smoothing_sigma)
        smoothed_ankle_y = gaussian_filter1d(data['lead_ankle_y'], sigma=smoothing_sigma)
        smoothed_knee_x = gaussian_filter1d(data['lead_knee_x'], sigma=smoothing_sigma)
        
        # Look for coordinated movement in multiple joints
        for i in range(baseline_length, len(smoothed_ankle_x) - 2):
            ankle_movement_x = abs(smoothed_ankle_x[i] - baseline_ankle_x)
            ankle_movement_y = abs(smoothed_ankle_y[i] - baseline_ankle_y)
            knee_movement_x = abs(smoothed_knee_x[i] - baseline_knee_x)
            
            # Stride involves coordinated ankle and knee movement
            total_movement = ankle_movement_x + ankle_movement_y + (knee_movement_x * 0.7)
            
            # Also check for movement acceleration (derivative)
            if i >= baseline_length + 2:
                movement_acceleration = (
                    abs(smoothed_ankle_x[i] - smoothed_ankle_x[i-2]) + 
                    abs(smoothed_ankle_y[i] - smoothed_ankle_y[i-2])
                ) / 2
                
                if (ankle_movement_x > threshold_x or ankle_movement_y > threshold_y or 
                    total_movement > threshold_x + threshold_y) and movement_acceleration > 0.03:
                    return data['frames'][i]
            
            elif ankle_movement_x > threshold_x or ankle_movement_y > threshold_y:
                return data['frames'][i]
        
        return None
    
    def _detect_foot_plant_adaptive(self, data, stride_start, video_length):
        """Adaptive foot plant detection using multi-signal analysis."""
        if stride_start is None or len(data['lead_ankle_y']) < 12:
            return None
        
        # Find stride start index with buffer
        start_idx = 0
        for i, frame in enumerate(data['frames']):
            if frame >= stride_start:
                start_idx = max(0, i - 1)  # Start slightly before for context
                break
        
        if start_idx >= len(data['lead_ankle_y']) - 6:
            return None
        
        # Adaptive smoothing based on video length
        smoothing_sigma = max(0.8, video_length / 100)
        smoothed_ankle_y = gaussian_filter1d(data['lead_ankle_y'], sigma=smoothing_sigma)
        smoothed_knee_y = gaussian_filter1d(data['lead_knee_y'], sigma=smoothing_sigma)
        
        # Multi-signal foot plant detection
        search_window = min(len(smoothed_ankle_y) - start_idx - 3, 
                           max(8, video_length // 4))
        
        for i in range(start_idx + 2, start_idx + search_window):
            if i >= len(smoothed_ankle_y) - 3:
                break
            
            # Ankle velocity analysis
            ankle_velocity = (smoothed_ankle_y[i+1] - smoothed_ankle_y[i-1]) / 2
            ankle_acceleration = (smoothed_ankle_y[i+2] - 2*smoothed_ankle_y[i] + smoothed_ankle_y[i-2]) / 4
            
            # Knee velocity for confirmation
            knee_velocity = (smoothed_knee_y[i+1] - smoothed_knee_y[i-1]) / 2
            
            # Foot plant: downward movement followed by deceleration/reversal
            # Plus knee movement slowing (weight acceptance)
            if (ankle_velocity < -0.02 and ankle_acceleration > 0.015 and 
                abs(knee_velocity) < 0.03):
                return data['frames'][i]
            
            # Alternative: sharp deceleration after movement
            if i > start_idx + 3:
                prev_velocity = (smoothed_ankle_y[i-1] - smoothed_ankle_y[i-3]) / 2
                velocity_change = abs(ankle_velocity - prev_velocity)
                
                if velocity_change > 0.04 and abs(ankle_velocity) < abs(prev_velocity) * 0.4:
                    return data['frames'][i]
        
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