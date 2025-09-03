from fastapi import FastAPI, Query
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
load_dotenv()  # Load variables from .env


AWS_REGION = os.getenv("AWS_REGION")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

app = FastAPI()

#  ----------------------  Upload to S3 Helper -----------------------

def upload_video_to_s3(file_path: str, s3_key: str) -> str:
    """
    Uploads a local file to S3 and returns a pre-signed URL valid for 1 hour.
    """
    try:
        # Validate environment variables
        required_vars = [AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME]
        if not all(required_vars):
            raise ValueError("Missing one or more AWS config values.")

        # Initialize S3 client
        s3 = boto3.client(
            "s3",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )

        print(f"📤 Uploading '{file_path}' to bucket '{S3_BUCKET_NAME}' as '{s3_key}'...")

        # Upload the file (no ACLs, just private upload)
        s3.upload_file(
            Filename=file_path,
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            ExtraArgs={"ContentType": "video/mp4"}
        )

        # Generate pre-signed URL valid for 1 hour
        presigned_url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": S3_BUCKET_NAME, "Key": s3_key},
            ExpiresIn=3600  # 1 hour in seconds
        )

        print(f"✅ Uploaded and generated pre-signed URL: {presigned_url}")
        return presigned_url

    except ValueError as ve:
        print(f"❌ AWS config error: {ve}")
        return ""
    except Exception as e:
        print(f"❌ Unexpected error during S3 upload: {e}")
        return ""


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

    # Upload to S3
    s3_key = f"swing-annotated/{uuid4()}.mp4"
    s3_url = upload_video_to_s3(video_output_path, s3_key)

    return {
        "feedback": feedback,
        "biomarkers": biomarker_results,
        "video_output_path": video_output_path,
        "s3_url": s3_url
    }

