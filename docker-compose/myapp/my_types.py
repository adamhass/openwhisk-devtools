from statefun import simple_type
import cv2
import base64
import numpy as np
import pickle

INT_BYTES = 4
## Images / Frames
def image_serializer(image, timestamp: int, camera: int):
    _, buffer = cv2.imencode('.jpg', image)
    serialised = base64.b64encode(buffer)
    bytes = timestamp.to_bytes(INT_BYTES, 'big') + \
        camera.to_bytes(INT_BYTES, 'big') + serialised
    return bytes

def image_deserializer(buf):
    timestamp = int.from_bytes(buf[0:INT_BYTES], 'big')
    camera = int.from_bytes(buf[INT_BYTES:2*INT_BYTES], 'big')
    jpg_original = base64.b64decode(buf[2*INT_BYTES:])
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    image = cv2.imdecode(jpg_as_np, flags=1)
    return (image, timestamp, camera)

## YOLOv3 Predictions
def pickle_serializer(data):
    buf = pickle.dumps(data)
    return buf

def pickle_deserializer(buf):
    data = pickle.loads(buf)
    return data

IMAGE = simple_type(
    typename='app/Image', serialize_fn=image_serializer, deserialize_fn=image_deserializer)

PREDICTION = simple_type(
    typename='app/Predictions', serialize_fn=pickle_serializer, deserialize_fn=pickle_deserializer)

BATCHES = simple_type(
    typename='app/Batches', serialize_fn=pickle_serializer, deserialize_fn=pickle_deserializer)
