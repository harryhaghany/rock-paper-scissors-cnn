# rock-paper-scissors-cnn
🪨📄✂️ Rock Paper Scissors Detector using CNN & Webcam

This project is a real-time Rock-Paper-Scissors (RPS) hand gesture classifier built using PyTorch, OpenCV, and a Convolutional Neural Network (CNN). The model is trained on a custom image dataset containing hand gestures for rock, paper, and scissors, and it uses your computer's webcam to recognize hand signs in live video.

🔍 What it Does
  -  Trains a CNN model on grayscale, normalized 28×28 images of hand gestures.
  -  Tests the trained model to evaluate its performance.
  -  Activates your webcam and detects RPS gestures from a live video feed in a central rectangular Region of Interest (ROI).
  -  Displays the predicted label ("Rock", "Paper", or "Scissors") in real-time.
    
📦 Features
  -  🔁 Real-Time Prediction using webcam feed.
  -  🧠 Custom CNN Model with two convolutional layers.
  -  📉 Training & Evaluation on a labeled image dataset.
  -  💻 User-Friendly interface via OpenCV display window.
  -  🎨 ROI (Region of Interest) centered for consistent classification.
    
🧠 Technologies Used
  -  PyTorch: Model building and training
  -  Torchvision: Image preprocessing and dataset handling
  -  OpenCV: Webcam capture and live visualization
  -  PIL: Image format conversion for preprocessing
