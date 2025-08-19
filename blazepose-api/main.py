from fastapi import FastAPI, Query
from robust_analyzer import RobustBaseballSwingAnalyzer, generate_annotated_video, analyze_video_from_url

app = FastAPI()

@app.get("/analyze/")
async def analyze(public_url: str = Query(...)):
    feedback, biomarker_results, rotated_path = await analyze_video_from_url(public_url)
    video_output_path = await generate_annotated_video(rotated_path, biomarker_results["swing_phases"])
    return {"feedback": feedback, "biomarkers": biomarker_results, "video_output_path": video_output_path}