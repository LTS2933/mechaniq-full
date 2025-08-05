from fastapi import FastAPI, Query
from analyzer import analyze_video_from_url, generate_annotated_video

app = FastAPI()

@app.get("/analyze/")
async def analyze(public_url: str = Query(...)):
    feedback, biomarker_results = await analyze_video_from_url(public_url)
    video_output_path = await generate_annotated_video(public_url, biomarker_results["swing_phases"])

    return {
        "feedback": feedback,
        "biomarkers": biomarker_results,
        "video_output_path": video_output_path  # Local path on your server
    }
