document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('video');
    const resultDiv = document.getElementById('result');
    const startBtn = document.getElementById('startBtn');
    let recognitionActive = false;
    let animationFrameId = null;

    // Camera setup
    const setupCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 }
            });
            video.srcObject = stream;
        } catch (err) {
            console.error("Camera error:", err);
            resultDiv.textContent = "Camera access denied";
        }
    };

    // Sign language mapping
    const labelMap = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
        5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
        10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P',
        15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U',
        20: 'V', 21: 'W', 22: 'X', 23: 'Y', 24: ' '
    };

    // Process predictions
    const processPrediction = (predictions) => {
        const maxPrediction = Math.max(...predictions);
        const predictedIndex = predictions.indexOf(maxPrediction);
        return {
            letter: labelMap[predictedIndex],
            confidence: (maxPrediction * 100).toFixed(2)
        };
    };

    // Main recognition loop
    const recognizeSign = async () => {
        if (!recognitionActive) return;

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: canvas.toDataURL('image/jpeg', 0.7)
                })
            });

            const data = await response.json();

            if (data.error) {
                resultDiv.textContent = `Error: ${data.error}`;
            } else if (data.prediction) {
                const { letter, confidence } = processPrediction(data.prediction);
                resultDiv.innerHTML = `Predicted: <strong>${letter}</strong> (${confidence}%)`;
            }
        } catch (err) {
            console.error("Prediction error:", err);
            resultDiv.textContent = "Prediction failed. See console for details.";
        }

        // Limit FPS to ~4 (250ms interval)
        setTimeout(() => {
            if (recognitionActive) recognizeSign();
        }, 250);
    };

    // Button control
    startBtn.addEventListener('click', () => {
        recognitionActive = !recognitionActive;
        startBtn.textContent = recognitionActive ? 'Stop' : 'Start';

        if (recognitionActive) {
            recognizeSign();
        }
    });

    // Initialize
    setupCamera();
});
