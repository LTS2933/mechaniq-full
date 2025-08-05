from fastapi import FastAPI, Query
from analyzer import analyze_video_from_url

app = FastAPI()

@app.get("/analyze/")
async def analyze(public_url: str = Query(...)):
    feedback = await analyze_video_from_url(public_url)
    return {"feedback": feedback}
