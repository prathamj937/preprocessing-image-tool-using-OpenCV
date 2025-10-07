import cv2
import numpy as np
import base64
from io import BytesIO
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def base64_to_image(base64_str):
    """Decode base64 string to OpenCV image"""
    header, encoded = base64_str.split(",", 1)
    image_data = base64.b64decode(encoded)
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def image_to_base64(image):
    """Convert OpenCV image (RGB) to base64 string"""
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", image)
    base64_str = base64.b64encode(buffer).decode("utf-8")
    return "data:image/jpeg;base64," + base64_str

def resize_image(image, scale):
    width = int(image.shape[1] * scale / 100)
    height = int(image.shape[0] * scale / 100)
    return cv2.resize(image, (width, height))

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def flip_image(image, mode):
    flip_code = 1 if mode == "horizontal" else 0 if mode == "vertical" else -1
    return cv2.flip(image, flip_code)

def apply_filter(image, filter_type, kernel_size):
    if filter_type == "gaussian_blur":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif filter_type == "median_blur":
        return cv2.medianBlur(image, kernel_size)
    elif filter_type == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
    return image

def edge_and_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    combined = np.hstack((gray, edges, binary, adaptive))
    return cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)

class ProcessRequest(BaseModel):
    action: str
    params: dict
    image: str

@app.post("/api/process")
async def process_image(request: ProcessRequest):
    try:
        image = base64_to_image(request.image)

        if request.action == "resize":
            scale = float(request.params.get("scale", 100))
            image = resize_image(image, scale)
        elif request.action == "rotate":
            angle = float(request.params.get("angle", 90))
            image = rotate_image(image, angle)
        elif request.action == "flip":
            mode = request.params.get("mode", "horizontal")
            image = flip_image(image, mode)
        elif request.action == "filter":
            ftype = request.params.get("type", "gaussian_blur")
            kernel = int(request.params.get("kernel_size", 5))
            image = apply_filter(image, ftype, kernel)
        elif request.action == "edge_threshold":
            image = edge_and_threshold(image)
        else:
            return {"success": False, "message": "Unknown action"}

        processed_base64 = image_to_base64(image)

        return {"success": True, "image": processed_base64}

    except Exception as e:
        print("Processing error:", e)
        return {"success": False, "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
