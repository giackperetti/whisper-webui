from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import uuid
import subprocess
import time

app = FastAPI()

# Directory to store uploads and outputs
UPLOAD_DIR = "/tmp/transcriptions"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Whisper model configuration
WHISPER_MODEL = "base"

@app.get("/")
async def hi():
    return "WhisperAI's API is up and running!"

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...), language: str = Form(...)):
    job_uuid = str(uuid.uuid4())
    filename = f"{job_uuid}_{file.filename}"
    local_audio_path = os.path.join(UPLOAD_DIR, filename)
    transcript_path = os.path.join(UPLOAD_DIR, f"{filename}.txt")

    # Save the uploaded file
    with open(local_audio_path, "wb") as f:
        f.write(await file.read())

    # Prepare the srun command
    srun_command = [
        "salloc", "-n 2", "-G 1", "bash",
        "srun", "-n 2", "-G 1", "--time=00:30:00", "--pty", "bash", "-i",
        f"""
        module load mamba &&
        source /opt/mambaforge/mamba/bin/activate &&
        conda activate whisper &&
        whisper --language {language} --model {WHISPER_MODEL} --output_format txt "{local_audio_path}" --output_dir {UPLOAD_DIR}
        """
    ]

    try:
        result = subprocess.run(
            srun_command,
            capture_output=True,
            text=True,
            check=True
        )

        timeout_seconds = 1800
        check_interval = 10
        elapsed = 0

        while elapsed < timeout_seconds:
            if os.path.exists(transcript_path):
                break
            time.sleep(check_interval)
            elapsed += check_interval
        else:
            raise HTTPException(status_code=504, detail="Transcription timed out")

        with open(transcript_path, "r") as f:
            transcript = f.read()

        return JSONResponse(content={
            "transcription": transcript,
            "job_id": job_uuid
        })

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"SLURM job execution failed: {e.stderr}")
