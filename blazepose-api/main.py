from fastapi import FastAPI, Query, HTTPException
from robust_analyzer import (
    RobustBaseballSwingAnalyzer,
    generate_annotated_video,
    analyze_video_from_url
)
import os
import boto3
from uuid import uuid4
from botocore.exceptions import BotoCoreError, NoCredentialsError, ClientError
from dotenv import load_dotenv
import subprocess

load_dotenv()  # Load variables from .env

AWS_REGION = os.getenv("AWS_REGION")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

app = FastAPI()

# ---------------------- Upload to S3 Helper -----------------------

def upload_video_to_s3(file_path: str, s3_key: str) -> str:
    try:
        required_vars = [AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME]
        if not all(required_vars):
            raise ValueError("Missing one or more AWS config values.")

        s3 = boto3.client(
            "s3",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )

        print(f"📤 Uploading '{file_path}' to bucket '{S3_BUCKET_NAME}' as '{s3_key}'...")

        s3.upload_file(
            Filename=file_path,
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            ExtraArgs={"ContentType": "video/mp4" if file_path.endswith(".mp4") else "image/jpeg"}
        )

        presigned_url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": S3_BUCKET_NAME, "Key": s3_key},
            ExpiresIn=3600
        )

        print(f"✅ Uploaded and generated pre-signed URL: {presigned_url}")
        return presigned_url

    except ValueError as ve:
        print(f"❌ AWS config error: {ve}")
        return ""
    except Exception as e:
        print(f"❌ Unexpected error during S3 upload: {e}")
        return ""

# ---------------------- Thumbnail Generator -----------------------

def generate_thumbnail(video_path: str, thumbnail_path: str, time_offset: str = "00:00:01"):
    try:
        command = [
            "ffmpeg",
            "-ss", time_offset,
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            thumbnail_path
        ]
        subprocess.run(command, check=True)
        print(f"🖼️ Thumbnail generated at {thumbnail_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to generate thumbnail: {e}")

# ---------------------- Analyze Endpoint -----------------------

@app.get("/analyze/")
async def analyze(public_url: str = Query(...)):
    feedback, biomarker_results, rotated_path, landmarks_by_frame, time_to_contact, hand_speed_metrics, attack_angle, hip_shoulder_separation = await analyze_video_from_url(public_url)

    handedness = biomarker_results.get("handedness", {}).get("handedness", "unknown")
    video_uuid = str(uuid4())

    # ---------- Generate annotated video ----------
    s3_key_video = f"swing-annotated/{video_uuid}.mp4"
    video_output_path = await generate_annotated_video(
        video_uuid=video_uuid,
        input_path=rotated_path,
        swing_phases=biomarker_results["swing_phases"],
        handedness=handedness,
        landmarks_by_frame=landmarks_by_frame,
        time_to_contact=time_to_contact,
        hand_speed_metrics=hand_speed_metrics,
        attack_angle=attack_angle,
        hip_shoulder_separation=hip_shoulder_separation
    )

    # ---------- Generate and upload thumbnail ----------
    thumbnail_path = f"{video_uuid}.jpg"
    generate_thumbnail(rotated_path, thumbnail_path)

    thumbnail_s3_key = f"thumbnails/{video_uuid}.jpg"
    thumbnail_url = upload_video_to_s3(thumbnail_path, thumbnail_s3_key)

    # ---------- Upload annotated video ----------
    video_s3_url = upload_video_to_s3(video_output_path, s3_key_video)

    # ---------- Cleanup local files ----------
    try:
        os.remove(video_output_path)
        print(f"🧹 Removed local video file: {video_output_path}")
    except Exception as e:
        print(f"⚠️ Could not delete video file: {e}")

    try:
        os.remove(thumbnail_path)
        print(f"🧹 Removed local thumbnail file: {thumbnail_path}")
    except Exception as e:
        print(f"⚠️ Could not delete thumbnail file: {e}")

    return {
        "feedback": feedback,
        "biomarkers": biomarker_results,
        "video_output_path": video_output_path,
        "s3_url": video_s3_url,
        "s3_key": s3_key_video,
        "uuid": video_uuid,
        "thumbnail_url": thumbnail_url,
        "thumbnail_s3_key": thumbnail_s3_key
    }

# ---------------------- Get Signed URL Endpoint -----------------------

@app.get("/get-signed-url/")
async def get_signed_s3_url(s3_key: str = Query(...)):
    try:
        if not all([AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME]):
            raise ValueError("Missing one or more AWS config values.")

        s3 = boto3.client(
            "s3",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )

        presigned_url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": S3_BUCKET_NAME, "Key": s3_key},
            ExpiresIn=60
        )

        return {"signed_url": presigned_url}

    except ClientError as e:
        print(f"❌ AWS S3 ClientError: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate signed URL")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
