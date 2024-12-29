# Live Emotion Detection Using Deep Learning

This project demonstrates a live emotion detection system using a Convolutional Neural Network (CNN). The model detects emotions in real-time from webcam video streams, identifying emotions like Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Features

- **Live Emotion Detection**: Real-time recognition of emotions from webcam input.
- **Custom Trained Model**: A CNN trained on the [FER-2013 Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge) for emotion classification.
- **Preprocessing**: Uses OpenCV for face detection and input preparation.
- **Model Deployment**: Runs a trained model to classify emotions with high accuracy.

---

## Installation

### Prerequisites

1. Python 3.6 or higher.
2. Install required Python libraries:
   ```bash
   pip install tensorflow opencv-python pandas numpy matplotlib
   ```

3. Download the pretrained model: [`emotion_recognition_model.h5`](#).
   - Place it in the same directory as the script.

### Clone the Repository

```bash
git clone https://github.com/<your-username>/live-emotion-detection.git
cd live-emotion-detection
```

---

## Usage

1. **Run the Live Emotion Detection Script:**

   ```bash
   python emotion_detection_live.py
   ```

2. **Training Your Own Model (Optional):**

   If you'd like to train your own model, ensure the FER-2013 dataset is downloaded and located at `D:\Python\Dataset Programs\Emotion Dataset\fer2013.csv`. Then run the training code provided in the repository.

---

## File Structure

- **`emotion_detection_live.py`**: Script for real-time emotion detection.
- **`emotion_recognition_model.h5`**: Pre-trained model for emotion recognition.
- **`train_model.py`**: Code for training the CNN model (if needed).

---

## Key Functions

### `emotion_detection_live.py`
- **Face Detection**: Uses OpenCV's Haar Cascade Classifier to detect faces in a video frame.
- **Emotion Prediction**: Loads the pre-trained model and predicts the emotion for each detected face.
- **Visualization**: Draws bounding boxes around faces and displays predicted emotions.

### `train_model.py`
- **Dataset Preparation**: Reads and preprocesses the FER-2013 dataset.
- **Model Definition**: Builds a CNN with Conv2D, MaxPooling2D, and Dense layers.
- **Training**: Trains the model and visualizes performance metrics.
- **Model Saving**: Saves the trained model for future use.

---

## Results

- **Accuracy**: Achieved a test accuracy of ~75% (subject to variation based on dataset and hyperparameters).
- **Real-Time Performance**: Processes frames efficiently for live emotion detection.

---

## Example Output

<img src="https://via.placeholder.com/600" alt="Live Emotion Detection" />

---

## Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [FER-2013 Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
- [OpenCV](https://opencv.org/)
- [TensorFlow](https://www.tensorflow.org/)
