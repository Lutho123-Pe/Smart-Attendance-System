import cv2
import numpy as np
from pathlib import Path


class ModelUnavailable(RuntimeError):
    pass


class FaceEngine:
    def __init__(self, detector_path, recognizer_path, threshold=0.40, margin=0.05):
        if not Path(detector_path).is_file() or not Path(recognizer_path).is_file():
            raise ModelUnavailable("Install the YuNet and SFace model files in the models/ directory first.")
        try:
            self.detector = cv2.FaceDetectorYN.create(
                detector_path, "", (320, 320), 0.9, 0.3, 5000
            )
            self.recognizer = cv2.FaceRecognizerSF.create(recognizer_path, "")
        except cv2.error as error:
            raise ModelUnavailable(f"Could not load the OpenCV face models: {error}") from error
        self.threshold = threshold
        self.margin = margin

    @staticmethod
    def _decode(image_bytes):
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError("The uploaded file is not a readable image.")
        height, width = image.shape[:2]
        if width > 4096 or height > 4096:
            raise ValueError("Images must be no larger than 4096 by 4096 pixels.")
        return image

    def _detect(self, image):
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        return [] if faces is None else faces

    def embedding(self, image_bytes):
        image = self._decode(image_bytes)
        faces = self._detect(image)
        if len(faces) != 1:
            raise ValueError("Each enrollment image must contain exactly one detectable face.")
        aligned = self.recognizer.alignCrop(image, faces[0])
        vector = self.recognizer.feature(aligned).flatten().astype(np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)

    @staticmethod
    def average(vectors):
        vector = np.mean(np.stack(vectors), axis=0).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError("Could not create a stable face template from these images.")
        return vector / norm

    def recognize(self, image_bytes, templates):
        image = self._decode(image_bytes)
        faces = self._detect(image)
        if len(faces) > 100:
            raise ValueError("Snapshot has too many faces to process safely (maximum 100).")
        results = []
        for face in faces:
            aligned = self.recognizer.alignCrop(image, face)
            vector = self.recognizer.feature(aligned).flatten().astype(np.float32)
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            scores = sorted(
                ((student_id, float(np.dot(vector, template))) for student_id, template in templates),
                key=lambda result: result[1],
                reverse=True,
            )
            student_id = None
            score = scores[0][1] if scores else 0.0
            second_score = scores[1][1] if len(scores) > 1 else -1.0
            if scores and score >= self.threshold and score - second_score >= self.margin:
                student_id = scores[0][0]
            results.append({"student_id": student_id, "score": score})
        return results