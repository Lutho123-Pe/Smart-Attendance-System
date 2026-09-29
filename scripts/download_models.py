from pathlib import Path
from urllib.request import urlopen


MODELS = {
    "face_detection_yunet_2023mar.onnx": "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def main():
    destination = Path(__file__).resolve().parents[1] / "models"
    destination.mkdir(exist_ok=True)
    for filename, url in MODELS.items():
        path = destination / filename
        if path.exists():
            print(f"Keeping existing {path}")
            continue
        print(f"Downloading {filename}")
        with urlopen(url, timeout=60) as response, path.open("wb") as model_file:
            model_file.write(response.read())
        print(f"Saved {path}")
    print("Review the OpenCV Zoo model licenses before use or redistribution.")


if __name__ == "__main__":
    main()