# %%writefile fiw_processor.py
import os
import pickle
import cv2
import numpy as np
from insightface.app import FaceAnalysis

def disk_writer_worker(queue):
    """This function lives in a file, so 'spawn' can find it."""
    while True:
        item = queue.get()
        if item is None:
            break
        path, payload = item
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(payload, f)
        except Exception as e:
            print(f"Error writing to {path}: {e}")

def get_face_app():
    # Force CoreML for your Mac Apple Silicon
    app = FaceAnalysis(name='buffalo_l', providers=['CoreMLExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_thresh=0.3, det_size=(640, 640))
    return app

def get_face_embedding(image_path, app):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Unable to read image at {image_path}")
            return None
        faces = app.get(img)
        if not faces:
            print(f"Warning: No faces detected in {image_path}")
            return None
        return faces[0].embedding
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None