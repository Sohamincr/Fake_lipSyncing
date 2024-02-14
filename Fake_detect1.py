from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import dlib
import librosa

app = Flask(__name__)

# Load facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to extract audio features
def extract_audio_features(audio_file):
    y, sr = librosa.load(audio_file)
    # Extract features (Example: MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # Calculate statistics over time
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    # Concatenate mean and standard deviation
    audio_features = np.concatenate((mfccs_mean, mfccs_std))
    return audio_features

# Function to detect lip landmarks
def detect_lip_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    if len(rects) == 0:
        return None

    # Assuming only one face is present in the image
    shape = predictor(gray, rects[0])

    # Extracting lip landmarks (landmarks 49-68 in the 68-point model)
    lip_landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(48, 68)]

    return lip_landmarks

# Function to evaluate correlation between facial landmarks and audio features
def evaluate_correlation(landmarks, audio_features):
    # Example: Calculate correlation between average y-coordinate of landmarks and MFCCs mean
    avg_landmark_y = np.mean([landmark[1] for landmark in landmarks])
    avg_mfccs_mean = np.mean(audio_features[:13])  # Assuming first 13 features are MFCCs mean
    correlation_measure = np.abs(avg_landmark_y - avg_mfccs_mean)
    return correlation_measure

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