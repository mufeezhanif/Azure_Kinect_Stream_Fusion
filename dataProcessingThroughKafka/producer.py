from confluent_kafka import Producer
import cv2
import sys
import numpy as np
import streamlit as st
from pyk4a import PyK4A, Config, ColorResolution, DepthMode
import pyk4a
import base64

def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")
        
# Configure the producer
conf = {
    'bootstrap.servers': 'localhost:9092',  # Replace with your Kafka server address
}

producer = Producer(conf)

def convertToBytes(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    return img_encoded.tobytes()

def colorize_depth_image(depth_image)-> np.array:
    depth_image: np.array = np.clip(depth_image, 0, 5000) 
    depth_image: np.array = (depth_image / 5000 * 255).astype(np.uint8) 
    return cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

def colorize_ir_image(ir_image) -> np.array:
    ir_image: np.array = np.clip(ir_image, 0, 500) 
    ir_image: np.array = (ir_image / 500 * 255).astype(np.uint8) 
    # return ir_image
    return cv2.applyColorMap(ir_image, cv2.COLORMAP_JET)

# setting device configurations
device_config = Config(
    color_resolution=pyk4a.ColorResolution.RES_1080P,
    depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,
    color_format = pyk4a.ImageFormat.COLOR_BGRA32,
)

device = PyK4A(config=device_config)
# Starting Device
device.start()
i = 0

while True:
    capture = device.get_capture()
    color_image = capture.color[:, :, :3]  
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2RGB)
    depth_image = colorize_depth_image(capture.depth)
    ir_image = colorize_ir_image(capture.ir)
    
    # encode images to bytes
    color_image = convertToBytes(color_image)
    depth_image= convertToBytes(depth_image)
    ir_image = convertToBytes(ir_image)
    
    # send images 
    producer.produce('colorImage', value=color_image, callback=delivery_report)
    producer.produce('depthImage', value=depth_image, callback=delivery_report)
    producer.produce('irImage', value=ir_image, callback=delivery_report)
    # clear the buffer 
    producer.flush()
    # for verification
    print(f'Frame {i} sent')
    i += 1

device.close()
device.stop()
print('Done')
