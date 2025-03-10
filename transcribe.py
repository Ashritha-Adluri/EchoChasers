import whisper
import subprocess
import os
from flask import Flask, render_template, request, send_from_directory
import warnings

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

app = Flask(__name__)

def convert_mp4_to_wav(mp4_file, wav_file):
    try:
        command = [
            'ffmpeg',
            '-i', mp4_file,
            '-vn',  
            '-acodec', 'pcm_s16le',  
            '-ar', '44100',  
            '-ac', '2', 
            wav_file
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Successfully converted {mp4_file} to {wav_file}")
    except subprocess.CalledProcessError:
        print(f"Error converting {mp4_file} to {wav_file}")
        raise

def transcribe_audio(audio_path):
    try:
        model = whisper.load_model("small")
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    input_file = os.path.join("uploads", file.filename)
    output_file = input_file.rsplit('.', 1)[0] + '.wav'

    file.save(input_file)

    if not os.path.exists(output_file):
        convert_mp4_to_wav(input_file, output_file)

    transcription = transcribe_audio(output_file)
    return render_template('result.html', transcription=transcription)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(debug=True)
