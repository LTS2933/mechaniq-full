from fastapi import FastAPI, Query
from robust_analyzer import (
    RobustBaseballSwingAnalyzer,
    generate_annotated_video,
    analyze_video_from_url
)

app = FastAPI()

@app.get("/analyze/")
async def analyze(public_url: str = Query(...)):
    feedback, biomarker_results, rotated_path, landmarks_by_frame, time_to_contact, hand_speed_metrics, attack_angle, hip_shoulder_separation = await analyze_video_from_url(public_url)

    handedness = biomarker_results.get("handedness", {}).get("handedness", "unknown")

    video_output_path = await generate_annotated_video(
        input_path=rotated_path,
        swing_phases=biomarker_results["swing_phases"],
        handedness=handedness,
        landmarks_by_frame=landmarks_by_frame,
        time_to_contact=time_to_contact,
        hand_speed_metrics=hand_speed_metrics,
        attack_angle=attack_angle,
        hip_shoulder_separation=hip_shoulder_separation
    )

    return {
        "feedback": feedback,
        "biomarkers": biomarker_results,
        "video_output_path": video_output_path  # e.g. 'annotated_output.mp4'
    }
