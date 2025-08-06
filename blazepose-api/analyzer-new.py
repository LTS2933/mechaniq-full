import cv2
import numpy as np
import mediapipe as mp
import tempfile
import httpx
import os
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple, Optional

class BaseballSwingAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def generate_annotated_video(
        self,
        original_video_path: str,
        output_video_path: str,
        landmarks_over_time: List,
        swing_phases: Dict,
        handedness: str
    ):
        """Generate a video showing pose landmarks and swing phases."""
        cap = cv2.VideoCapture(original_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        font = cv2.FONT_HERSHEY_SIMPLEX

        i = 0
        phase_labels = {v['frame']: k for k, v in swing_phases.items() if v['frame'] is not None}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or i >= len(landmarks_over_time):
                break

            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                mp.solutions.pose.PoseLandmarkList(landmark=landmarks_over_time[i]),
                self.mp_pose.POSE_CONNECTIONS
            )

            # Overlay swing phase
            if i in phase_labels:
                label = phase_labels[i].replace("_", " ").title()
                cv2.putText(frame, f"Phase: {label}", (50, 50), font, 1, (0, 255, 0), 2)

            # Overlay handedness
            cv2.putText(frame, f"Handedness: {handedness}", (50, height - 30), font, 1, (255, 255, 255), 2)

            out.write(frame)
            i += 1

        cap.release()
        out.release()

    
    def calculate_velocity(self, positions: List[float], fps: float = 30.0) -> List[float]:
        """Calculate velocity from position data"""
        if len(positions) < 2:
            return [0.0]
        
        velocities = []
        dt = 1.0 / fps
        
        for i in range(len(positions)):
            if i == 0:
                vel = (positions[1] - positions[0]) / dt
            elif i == len(positions) - 1:
                vel = (positions[i] - positions[i-1]) / dt
            else:
                vel = (positions[i+1] - positions[i-1]) / (2 * dt)
            velocities.append(vel)
        
        return velocities
    
    def calculate_acceleration(self, velocities: List[float], fps: float = 30.0) -> List[float]:
        """Calculate acceleration from velocity data"""
        if len(velocities) < 2:
            return [0.0]
        
        accelerations = []
        dt = 1.0 / fps
        
        for i in range(len(velocities)):
            if i == 0:
                acc = (velocities[1] - velocities[0]) / dt
            elif i == len(velocities) - 1:
                acc = (velocities[i] - velocities[i-1]) / dt
            else:
                acc = (velocities[i+1] - velocities[i-1]) / (2 * dt)
            accelerations.append(acc)
        
        return accelerations
    
    def smooth_data(self, data: List[float], sigma: float = 1.0) -> List[float]:
        """Apply Gaussian smoothing to data"""
        if len(data) < 3:
            return data
        return gaussian_filter1d(data, sigma=sigma).tolist()
    
    def enhanced_handedness_detection(self, landmarks_over_time: List, fps: float = 30.0) -> Dict:
        """
        Enhanced handedness detection using multiple factors:
        1. Shoulder positioning analysis
        2. Elbow angle analysis
        3. Stance width and foot positioning
        4. Initial body orientation
        5. Movement pattern analysis
        """
        results = {
            'handedness': 'unknown',
            'confidence': 0.0,
            'factors': {},
            'debug_info': {}
        }
        
        if len(landmarks_over_time) < 10:
            return results
        
        # Factor 1: Shoulder positioning analysis
        shoulder_votes = self._analyze_shoulder_positioning(landmarks_over_time)
        
        # Factor 2: Elbow angle analysis
        elbow_votes = self._analyze_elbow_angles(landmarks_over_time)
        
        # Factor 3: Stance analysis
        stance_votes = self._analyze_stance(landmarks_over_time)
        
        # Factor 4: Movement pattern analysis
        movement_votes = self._analyze_movement_patterns(landmarks_over_time, fps)
        
        # Factor 5: Hip rotation analysis
        hip_votes = self._analyze_hip_rotation(landmarks_over_time)
        
        # Combine all factors with weights
        factors = {
            'shoulder_positioning': shoulder_votes,
            'elbow_angles': elbow_votes,
            'stance_analysis': stance_votes,
            'movement_patterns': movement_votes,
            'hip_rotation': hip_votes
        }
        
        # Weight the factors
        weights = {
            'shoulder_positioning': 0.25,
            'elbow_angles': 0.20,
            'stance_analysis': 0.20,
            'movement_patterns': 0.20,
            'hip_rotation': 0.15
        }
        
        left_score = 0.0
        right_score = 0.0
        total_weight = 0.0
        
        for factor, vote in factors.items():
            if vote['confidence'] > 0.3:  # Only use confident votes
                weight = weights[factor] * vote['confidence']
                if vote['handedness'] == 'left':
                    left_score += weight
                elif vote['handedness'] == 'right':
                    right_score += weight
                total_weight += weight
        
        # Determine final handedness
        if total_weight > 0:
            if left_score > right_score and left_score > total_weight * 0.6:
                results['handedness'] = 'left'
                results['confidence'] = left_score / total_weight
            elif right_score > left_score and right_score > total_weight * 0.6:
                results['handedness'] = 'right'
                results['confidence'] = right_score / total_weight
        
        results['factors'] = factors
        results['debug_info'] = {
            'left_score': left_score,
            'right_score': right_score,
            'total_weight': total_weight
        }
        
        return results
    
    def _analyze_shoulder_positioning(self, landmarks_over_time: List) -> Dict:
        """Analyze shoulder positioning relative to screen"""
        left_votes = 0
        right_votes = 0
        total_frames = 0
        
        for lm in landmarks_over_time[:20]:  # Use first 20 frames
            try:
                left_shoulder = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                
                # In a proper batting stance from side view:
                # Right-handed: left shoulder further from camera (higher x)
                # Left-handed: right shoulder further from camera (higher x)
                
                if left_shoulder.x > right_shoulder.x:
                    right_votes += 1  # Left shoulder back = right handed
                else:
                    left_votes += 1   # Right shoulder back = left handed
                
                total_frames += 1
            except:
                continue
        
        if total_frames < 5:
            return {'handedness': 'unknown', 'confidence': 0.0}
        
        confidence = max(left_votes, right_votes) / total_frames
        handedness = 'right' if right_votes > left_votes else 'left'
        
        return {'handedness': handedness, 'confidence': confidence}
    
    def _analyze_elbow_angles(self, landmarks_over_time: List) -> Dict:
        """Analyze elbow angles and positioning"""
        left_votes = 0
        right_votes = 0
        total_frames = 0
        
        for lm in landmarks_over_time[:15]:
            try:
                left_shoulder = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                left_elbow = lm[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
                left_wrist = lm[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
                
                right_shoulder = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                right_elbow = lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                right_wrist = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
                
                # Calculate elbow angles
                left_angle = self._calculate_angle(
                    [left_shoulder.x, left_shoulder.y],
                    [left_elbow.x, left_elbow.y],
                    [left_wrist.x, left_wrist.y]
                )
                
                right_angle = self._calculate_angle(
                    [right_shoulder.x, right_shoulder.y],
                    [right_elbow.x, right_elbow.y],
                    [right_wrist.x, right_wrist.y]
                )
                
                # In batting stance, top hand typically has more acute angle
                if left_angle < right_angle - 10:  # Left elbow more bent
                    right_votes += 1  # Left hand on top = right handed
                elif right_angle < left_angle - 10:  # Right elbow more bent
                    left_votes += 1   # Right hand on top = left handed
                
                total_frames += 1
            except:
                continue
        
        if total_frames < 3:
            return {'handedness': 'unknown', 'confidence': 0.0}
        
        confidence = max(left_votes, right_votes) / total_frames if total_frames > 0 else 0.0
        handedness = 'right' if right_votes > left_votes else 'left'
        
        return {'handedness': handedness, 'confidence': confidence}
    
    def _analyze_stance(self, landmarks_over_time: List) -> Dict:
        """Analyze batting stance and foot positioning"""
        left_votes = 0
        right_votes = 0
        total_frames = 0
        
        for lm in landmarks_over_time[:10]:
            try:
                left_ankle = lm[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
                right_ankle = lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                left_hip = lm[self.mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
                
                # Analyze hip-ankle alignment
                left_alignment = abs(left_hip.x - left_ankle.x)
                right_alignment = abs(right_hip.x - right_ankle.x)
                
                # In proper stance, back foot typically shows more offset
                if left_alignment > right_alignment + 0.01:
                    right_votes += 1  # Left foot back = right handed
                elif right_alignment > left_alignment + 0.01:
                    left_votes += 1   # Right foot back = left handed
                
                total_frames += 1
            except:
                continue
        
        if total_frames < 3:
            return {'handedness': 'unknown', 'confidence': 0.0}
        
        confidence = max(left_votes, right_votes) / total_frames if total_frames > 0 else 0.0
        handedness = 'right' if right_votes > left_votes else 'left'
        
        return {'handedness': handedness, 'confidence': confidence}
    
    def _analyze_movement_patterns(self, landmarks_over_time: List, fps: float) -> Dict:
        """Analyze initial movement patterns to determine handedness"""
        if len(landmarks_over_time) < 20:
            return {'handedness': 'unknown', 'confidence': 0.0}
        
        # Track ankle movements
        left_ankle_x = []
        right_ankle_x = []
        
        for lm in landmarks_over_time:
            try:
                left_ankle_x.append(lm[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x)
                right_ankle_x.append(lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x)
            except:
                continue
        
        if len(left_ankle_x) < 10:
            return {'handedness': 'unknown', 'confidence': 0.0}
        
        # Calculate velocities
        left_vel = self.calculate_velocity(left_ankle_x, fps)
        right_vel = self.calculate_velocity(right_ankle_x, fps)
        
        # Find first significant movement
        movement_threshold = 0.1  # Adjust based on video resolution
        
        left_movement_frame = None
        right_movement_frame = None
        
        for i in range(len(left_vel)):
            if left_movement_frame is None and abs(left_vel[i]) > movement_threshold:
                left_movement_frame = i
            if right_movement_frame is None and abs(right_vel[i]) > movement_threshold:
                right_movement_frame = i
        
        confidence = 0.0
        handedness = 'unknown'
        
        if left_movement_frame is not None and right_movement_frame is not None:
            if left_movement_frame < right_movement_frame - 2:
                handedness = 'right'  # Left foot moves first = right handed
                confidence = 0.8
            elif right_movement_frame < left_movement_frame - 2:
                handedness = 'left'   # Right foot moves first = left handed
                confidence = 0.8
        elif left_movement_frame is not None and right_movement_frame is None:
            handedness = 'right'
            confidence = 0.6
        elif right_movement_frame is not None and left_movement_frame is None:
            handedness = 'left'
            confidence = 0.6
        
        return {'handedness': handedness, 'confidence': confidence}
    
    def _analyze_hip_rotation(self, landmarks_over_time: List) -> Dict:
        """Analyze hip rotation patterns"""
        hip_angles = []
        
        for lm in landmarks_over_time[:15]:
            try:
                left_hip = lm[self.mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
                
                # Calculate hip angle relative to horizontal
                angle = np.degrees(np.arctan2(right_hip.y - left_hip.y, right_hip.x - left_hip.x))
                hip_angles.append(angle)
            except:
                continue
        
        if len(hip_angles) < 5:
            return {'handedness': 'unknown', 'confidence': 0.0}
        
        # Analyze initial hip orientation
        initial_angle = np.mean(hip_angles[:5])
        
        # In typical batting stance:
        # Right-handed: hips slightly open (positive angle)
        # Left-handed: hips slightly closed (negative angle)
        
        confidence = min(abs(initial_angle) / 15.0, 1.0)  # Normalize confidence
        
        if initial_angle > 5:
            handedness = 'right'
        elif initial_angle < -5:
            handedness = 'left'
        else:
            handedness = 'unknown'
            confidence = 0.0
        
        return {'handedness': handedness, 'confidence': confidence}
    
    def _calculate_angle(self, a: List[float], b: List[float], c: List[float]) -> float:
        """Calculate angle between three points"""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180 / np.pi)
        return 360 - angle if angle > 180 else angle
    
    def enhanced_swing_phase_detection(self, landmarks_over_time: List, handedness: str, fps: float = 30.0) -> Dict:
        """
        Enhanced swing phase detection with confidence scoring and velocity analysis
        """
        phases = {
            "stride_start": {"frame": None, "confidence": 0.0},
            "foot_plant": {"frame": None, "confidence": 0.0},
            "swing_start": {"frame": None, "confidence": 0.0},
            "contact": {"frame": None, "confidence": 0.0},
            "follow_through": {"frame": None, "confidence": 0.0},
        }
        
        if len(landmarks_over_time) < 10:
            return phases
        
        is_lefty = handedness == 'left'
        
        # Extract position data
        position_data = self._extract_position_data(landmarks_over_time, is_lefty)
        
        # Calculate velocities and accelerations
        velocity_data = self._calculate_velocity_data(position_data, fps)
        acceleration_data = self._calculate_acceleration_data(velocity_data, fps)
        
        # Detect each phase
        phases["stride_start"] = self._detect_stride_start(position_data, velocity_data, is_lefty)
        phases["foot_plant"] = self._detect_foot_plant(position_data, velocity_data, acceleration_data, is_lefty, phases["stride_start"]["frame"])
        phases["swing_start"] = self._detect_swing_start(position_data, velocity_data, is_lefty, phases["foot_plant"]["frame"])
        phases["contact"] = self._detect_contact(position_data, velocity_data, is_lefty, phases["swing_start"]["frame"])
        phases["follow_through"] = self._detect_follow_through(position_data, velocity_data, is_lefty, phases["contact"]["frame"])
        
        return phases
    
    def _extract_position_data(self, landmarks_over_time: List, is_lefty: bool) -> Dict:
        """Extract relevant position data for analysis"""
        data = {
            'lead_ankle_x': [], 'lead_ankle_y': [],
            'back_ankle_x': [], 'back_ankle_y': [],
            'lead_knee_x': [], 'lead_knee_y': [],
            'back_knee_x': [], 'back_knee_y': [],
            'lead_hip_x': [], 'lead_hip_y': [],
            'back_hip_x': [], 'back_hip_y': [],
            'lead_shoulder_x': [], 'lead_shoulder_y': [],
            'back_shoulder_x': [], 'back_shoulder_y': [],
            'lead_elbow_x': [], 'lead_elbow_y': [],
            'back_elbow_x': [], 'back_elbow_y': [],
            'lead_wrist_x': [], 'lead_wrist_y': [],
            'back_wrist_x': [], 'back_wrist_y': []
        }
        
        # Define lead/back based on handedness
        lead_side = 'RIGHT' if is_lefty else 'LEFT'
        back_side = 'LEFT' if is_lefty else 'RIGHT'
        
        for lm in landmarks_over_time:
            try:
                # Ankles
                lead_ankle = lm[getattr(self.mp_pose.PoseLandmark, f'{lead_side}_ANKLE').value]
                back_ankle = lm[getattr(self.mp_pose.PoseLandmark, f'{back_side}_ANKLE').value]
                data['lead_ankle_x'].append(lead_ankle.x)
                data['lead_ankle_y'].append(lead_ankle.y)
                data['back_ankle_x'].append(back_ankle.x)
                data['back_ankle_y'].append(back_ankle.y)
                
                # Knees
                lead_knee = lm[getattr(self.mp_pose.PoseLandmark, f'{lead_side}_KNEE').value]
                back_knee = lm[getattr(self.mp_pose.PoseLandmark, f'{back_side}_KNEE').value]
                data['lead_knee_x'].append(lead_knee.x)
                data['lead_knee_y'].append(lead_knee.y)
                data['back_knee_x'].append(back_knee.x)
                data['back_knee_y'].append(back_knee.y)
                
                # Hips
                lead_hip = lm[getattr(self.mp_pose.PoseLandmark, f'{lead_side}_HIP').value]
                back_hip = lm[getattr(self.mp_pose.PoseLandmark, f'{back_side}_HIP').value]
                data['lead_hip_x'].append(lead_hip.x)
                data['lead_hip_y'].append(lead_hip.y)
                data['back_hip_x'].append(back_hip.x)
                data['back_hip_y'].append(back_hip.y)
                
                # Shoulders
                lead_shoulder = lm[getattr(self.mp_pose.PoseLandmark, f'{lead_side}_SHOULDER').value]
                back_shoulder = lm[getattr(self.mp_pose.PoseLandmark, f'{back_side}_SHOULDER').value]
                data['lead_shoulder_x'].append(lead_shoulder.x)
                data['lead_shoulder_y'].append(lead_shoulder.y)
                data['back_shoulder_x'].append(back_shoulder.x)
                data['back_shoulder_y'].append(back_shoulder.y)
                
                # Elbows
                lead_elbow = lm[getattr(self.mp_pose.PoseLandmark, f'{lead_side}_ELBOW').value]
                back_elbow = lm[getattr(self.mp_pose.PoseLandmark, f'{back_side}_ELBOW').value]
                data['lead_elbow_x'].append(lead_elbow.x)
                data['lead_elbow_y'].append(lead_elbow.y)
                data['back_elbow_x'].append(back_elbow.x)
                data['back_elbow_y'].append(back_elbow.y)
                
                # Wrists
                lead_wrist = lm[getattr(self.mp_pose.PoseLandmark, f'{lead_side}_WRIST').value]
                back_wrist = lm[getattr(self.mp_pose.PoseLandmark, f'{back_side}_WRIST').value]
                data['lead_wrist_x'].append(lead_wrist.x)
                data['lead_wrist_y'].append(lead_wrist.y)
                data['back_wrist_x'].append(back_wrist.x)
                data['back_wrist_y'].append(back_wrist.y)
                
            except Exception as e:
                # Fill with previous value or zero if first frame
                for key in data.keys():
                    if len(data[key]) > 0:
                        data[key].append(data[key][-1])
                    else:
                        data[key].append(0.0)
        
        return data
    
    def _calculate_velocity_data(self, position_data: Dict, fps: float) -> Dict:
        """Calculate velocity data for all tracked points"""
        velocity_data = {}
        for key, positions in position_data.items():
            velocity_data[key.replace('_x', '_vel_x').replace('_y', '_vel_y')] = self.calculate_velocity(positions, fps)
        return velocity_data
    
    def _calculate_acceleration_data(self, velocity_data: Dict, fps: float) -> Dict:
        """Calculate acceleration data for all tracked points"""
        acceleration_data = {}
        for key, velocities in velocity_data.items():
            acceleration_data[key.replace('_vel_', '_acc_')] = self.calculate_acceleration(velocities, fps)
        return acceleration_data
    
    def _detect_stride_start(self, position_data: Dict, velocity_data: Dict, is_lefty: bool) -> Dict:
        """Detect stride start using velocity analysis"""
        lead_ankle_vel = velocity_data.get('lead_ankle_vel_x', [])
        lead_knee_vel = velocity_data.get('lead_knee_vel_x', [])
        
        if len(lead_ankle_vel) < 5:
            return {"frame": None, "confidence": 0.0}
        
        # Look for sustained movement in lead leg
        movement_threshold = 0.02
        confidence_threshold = 0.7
        
        for i in range(2, len(lead_ankle_vel) - 2):
            # Check for sustained movement over 3-5 frames
            ankle_movement = np.mean(np.abs(lead_ankle_vel[i:i+3]))
            knee_movement = np.mean(np.abs(lead_knee_vel[i:i+3]))
            
            if ankle_movement > movement_threshold or knee_movement > movement_threshold:
                # Calculate confidence based on movement consistency
                consistency = 1.0 - np.std(lead_ankle_vel[i:i+3]) / (ankle_movement + 1e-6)
                confidence = min(consistency * (ankle_movement / movement_threshold), 1.0)
                
                if confidence > confidence_threshold:
                    return {"frame": i, "confidence": confidence}
        
        return {"frame": None, "confidence": 0.0}
    
    def _detect_foot_plant(self, position_data: Dict, velocity_data: Dict, acceleration_data: Dict, is_lefty: bool, stride_start_frame: Optional[int]) -> Dict:
        """Detect foot plant using deceleration analysis"""
        if stride_start_frame is None:
            search_start = 5
        else:
            search_start = stride_start_frame + 3
        
        lead_ankle_vel = velocity_data.get('lead_ankle_vel_x', [])
        lead_ankle_acc = acceleration_data.get('lead_ankle_acc_x', [])
        
        if len(lead_ankle_vel) < search_start + 5:
            return {"frame": None, "confidence": 0.0}
        
        # Look for deceleration (negative acceleration) after movement
        for i in range(search_start, len(lead_ankle_acc) - 2):
            # Check for strong deceleration
            deceleration = -lead_ankle_acc[i]  # Negative because we want slowing down
            velocity_magnitude = abs(lead_ankle_vel[i])
            
            if deceleration > 0.1 and velocity_magnitude < 0.05:  # Slowing down significantly
                confidence = min(deceleration / 0.2, 1.0)  # Normalize confidence
                return {"frame": i, "confidence": confidence}
        
        return {"frame": None, "confidence": 0.0}
    
    def _detect_swing_start(self, position_data: Dict, velocity_data: Dict, is_lefty: bool, foot_plant_frame: Optional[int]) -> Dict:
        """Detect swing start using hand/wrist velocity"""
        if foot_plant_frame is None:
            search_start = 10
        else:
            search_start = foot_plant_frame + 1
        
        back_wrist_vel_x = velocity_data.get('back_wrist_vel_x', [])
        back_elbow_vel_x = velocity_data.get('back_elbow_vel_x', [])
        
        if len(back_wrist_vel_x) < search_start + 5:
            return {"frame": None, "confidence": 0.0}
        
        # Look for forward acceleration of back hand
        movement_threshold = 0.03
        
        for i in range(search_start, len(back_wrist_vel_x) - 2):
            wrist_vel = back_wrist_vel_x[i]
            elbow_vel = back_elbow_vel_x[i] if i < len(back_elbow_vel_x) else 0
            
            # Check for forward movement (positive velocity for back hand)
            if wrist_vel > movement_threshold and elbow_vel > movement_threshold * 0.5:
                confidence = min(wrist_vel / (movement_threshold * 2), 1.0)
                return {"frame": i, "confidence": confidence}
        
        return {"frame": None, "confidence": 0.0}
    
    def _detect_contact(self, position_data: Dict, velocity_data: Dict, is_lefty: bool, swing_start_frame: Optional[int]) -> Dict:
        """Detect contact point using maximum velocity"""
        if swing_start_frame is None:
            search_start = 15
        else:
            search_start = swing_start_frame + 2
        
        back_wrist_vel_x = velocity_data.get('back_wrist_vel_x', [])
        lead_wrist_vel_x = velocity_data.get('lead_wrist_vel_x', [])
        
        if len(back_wrist_vel_x) < search_start + 5:
            return {"frame": None, "confidence": 0.0}
        
        # Find maximum velocity point (indicates contact)
        max_vel = 0
        max_frame = None
        
        search_end = min(search_start + 15, len(back_wrist_vel_x))
        
        for i in range(search_start, search_end):
            combined_vel = abs(back_wrist_vel_x[i])
            if i < len(lead_wrist_vel_x):
                combined_vel += abs(lead_wrist_vel_x[i])
            
            if combined_vel > max_vel:
                max_vel = combined_vel
                max_frame = i
        
        if max_frame is not None and max_vel > 0.05:
            confidence = min(max_vel / 0.2, 1.0)
            return {"frame": max_frame, "confidence": confidence}
        
        return {"frame": None, "confidence": 0.0}
    
    def _detect_follow_through(self, position_data: Dict, velocity_data: Dict, is_lefty: bool, contact_frame: Optional[int]) -> Dict:
        """Detect follow through using velocity decay"""
        if contact_frame is None:
            search_start = 20
        else:
            search_start = contact_frame + 2
        
        back_wrist_vel_x = velocity_data.get('back_wrist_vel_x', [])
        
        if len(back_wrist_vel_x) < search_start + 5:
            return {"frame": None, "confidence": 0.0}
        
        # Look for significant velocity reduction
        initial_vel = abs(back_wrist_vel_x[search_start]) if search_start < len(back_wrist_vel_x) else 0
        
        for i in range(search_start + 1, min(search_start + 10, len(back_wrist_vel_x))):
            current_vel = abs(back_wrist_vel_x[i])
            
            # Check if velocity has dropped significantly
            if initial_vel > 0 and current_vel < initial_vel * 0.3:
                confidence = (initial_vel - current_vel) / initial_vel
                return {"frame": i, "confidence": min(confidence, 1.0)}
        
        return {"frame": None, "confidence": 0.0}
    
    def calculate_separation_angles(self, landmarks_over_time: List, handedness: str) -> List[Dict]:
        """Calculate hip-shoulder separation angles with enhanced accuracy"""
        separation_data = []
        is_lefty = handedness == 'left'
        
        for i, lm in enumerate(landmarks_over_time):
            try:
                # Get hip landmarks
                left_hip = lm[self.mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
                
                # Get shoulder landmarks
                left_shoulder = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                
                # Calculate hip angle
                hip_angle = np.degrees(np.arctan2(right_hip.y - left_hip.y, right_hip.x - left_hip.x))
                
                # Calculate shoulder angle
                shoulder_angle = np.degrees(np.arctan2(right_shoulder.y - left_shoulder.y, right_shoulder.x - left_shoulder.x))
                
                # Calculate separation (difference between shoulder and hip rotation)
                separation = abs((shoulder_angle - hip_angle + 180) % 360 - 180)
                
                separation_data.append({
                    'frame': i,
                    'hip_angle': hip_angle,
                    'shoulder_angle': shoulder_angle,
                    'separation': separation
                })
                
            except Exception as e:
                separation_data.append({
                    'frame': i,
                    'hip_angle': 0,
                    'shoulder_angle': 0,
                    'separation': 0
                })
        
        return separation_data
    
    async def analyze_video_from_url(self, url: str) -> Dict:
        """Analyze video from URL with enhanced algorithms"""
        try:
            # Download video
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                if response.status_code != 200:
                    return {"error": "Failed to download video", "status": "error"}
                
                video_bytes = response.content
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_bytes)
                video_path = tmp.name
            
            # Analyze the video
            result = await self.analyze_video_file(video_path)
            
            # Clean up
            os.remove(video_path)
            
            return result
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}", "status": "error"}
    
    async def analyze_video_file(self, video_path: str) -> Dict:
        """Analyze video file with enhanced algorithms"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": "Failed to open video file", "status": "error"}
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = 0
            landmarks_over_time = []
            
            # Process video frames
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(image_rgb)
                
                if results.pose_landmarks:
                    landmarks_over_time.append(results.pose_landmarks.landmark)
                
                frame_count += 1
                
                # Limit processing for performance
                if frame_count > 300:  # About 10 seconds at 30fps
                    break
            
            cap.release()
            
            if len(landmarks_over_time) < 10:
                return {"error": "Insufficient pose data detected", "status": "error"}

            output_video_path = "analyzed_output.mp4"
            self.generate_annotated_video(
                original_video_path=video_path,
                output_video_path=output_video_path,
                landmarks_over_time=landmarks_over_time,
                swing_phases=swing_phases,
                handedness=handedness_result['handedness']
            )
            
            # Enhanced handedness detection
            handedness_result = self.enhanced_handedness_detection(landmarks_over_time, fps)
            
            # Enhanced swing phase detection
            swing_phases = self.enhanced_swing_phase_detection(
                landmarks_over_time, 
                handedness_result['handedness'], 
                fps
            )
            
            # Calculate separation angles
            separation_data = self.calculate_separation_angles(
                landmarks_over_time, 
                handedness_result['handedness']
            )
            
            # Find peak separation
            max_separation = max(separation_data, key=lambda x: x['separation']) if separation_data else None
            
            # Compile results
            result = {
                "status": "success",
                "handedness": handedness_result,
                "swing_phases": swing_phases,
                "separation_analysis": {
                    "data": separation_data,
                    "max_separation": max_separation,
                    "average_separation": np.mean([d['separation'] for d in separation_data]) if separation_data else 0
                },
                "video_info": {
                    "fps": fps,
                    "frames_analyzed": len(landmarks_over_time),
                    "duration_seconds": len(landmarks_over_time) / fps
                },
                "feedback": self._generate_feedback(handedness_result, swing_phases, separation_data)
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}", "status": "error"}
    
    def _generate_feedback(self, handedness_result: Dict, swing_phases: Dict, separation_data: List[Dict]) -> List[Dict]:
        """Generate coaching feedback based on analysis"""
        feedback = []
        
        # Handedness confidence feedback
        if handedness_result['confidence'] < 0.7:
            feedback.append({
                "type": "warning",
                "message": f"Handedness detection confidence is low ({handedness_result['confidence']:.1%}). Consider recording from a clearer side angle."
            })
        
        # Swing timing feedback
        missing_phases = [phase for phase, data in swing_phases.items() if data['frame'] is None]
        if missing_phases:
            feedback.append({
                "type": "info",
                "message": f"Could not detect these swing phases: {', '.join(missing_phases)}. This may indicate incomplete swing or poor video angle."
            })
        
        # Separation feedback
        if separation_data:
            max_sep = max(d['separation'] for d in separation_data)
            avg_sep = np.mean([d['separation'] for d in separation_data])
            
            if max_sep < 20:
                feedback.append({
                    "type": "coaching",
                    "message": f"Low hip-shoulder separation (max: {max_sep:.1f}°). Work on creating more torque by rotating hips first."
                })
            elif max_sep > 60:
                feedback.append({
                    "type": "coaching",
                    "message": f"Very high hip-shoulder separation (max: {max_sep:.1f}°). Good separation, but ensure you can still make contact consistently."
                })
            else:
                feedback.append({
                    "type": "positive",
                    "message": f"Good hip-shoulder separation (max: {max_sep:.1f}°). This creates good power potential."
                })
        
        # Phase confidence feedback
        low_confidence_phases = [phase for phase, data in swing_phases.items() 
                               if data['frame'] is not None and data['confidence'] < 0.6]
        
        if low_confidence_phases:
            feedback.append({
                "type": "info",
                "message": f"Low confidence in detecting: {', '.join(low_confidence_phases)}. Consider recording with better lighting and clearer view."
            })
        
        return feedback
