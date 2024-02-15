from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import dlib
import librosa
from Fake_detect1 import extract_audio_features,detect_lip_landmarks,evaluate_correlation

app = Flask(__name__)
#@app.route('/', methods=['GET'])


#@app.route('/lip_sync_detection', methods=['POST'])
def lip_sync_detection(audio_file,video_file):

    # Extract audio features
    audio_features = extract_audio_features(audio_file)

    # Process video frames
    video_capture = cv2.VideoCapture(video_file)
    response = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect lip landmarks in each frame
        landmarks = detect_lip_landmarks(frame)

        if landmarks is not None:
            # Evaluate correlation between facial landmarks and audio features
            correlation = evaluate_correlation(landmarks, audio_features)

            # Make decision based on correlation measure
            if correlation > 395:
                decision = "Fake"
            else:
                decision = "Genuine"
            print(decision)
            response.append({"frame": frame, "decision": decision})

    # Release resources
    video_capture.release()

    return jsonify(response)



 # Receive files from request
audio_file = "output_audio20.mp3"
video_file = "Rec-65b4d436e01621cac2c1e718-1706349719530.mp4"

output = lip_sync_detection(audio_file,video_file)

print(output)

if __name__ == '__main__':
    app.run(debug=True)
