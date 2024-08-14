import streamlit as st
import numpy as np
import cv2
import base64
from confluent_kafka import Consumer, KafkaException
import sys


def bytes_to_image(image_bytes: np.ndarray)->np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'my_consumer_group',
    'auto.offset.reset': 'earliest',
}
#creating separate consumers 
rgConsumer = Consumer(conf)
rgConsumer.subscribe(['colorImage'])
irConsumer = Consumer(conf)
irConsumer.subscribe(['depthImage'])
depthConsumer = Consumer(conf)
depthConsumer.subscribe(['irImage'])

# encode image in base64 string
def imgAtBase64(file):
    with open(file,"rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = imgAtBase64('assets/bg.jpg')
img1 = imgAtBase64('assets/bgs.jpg')

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img1}");
background-size: 100%;
backfround-position: static;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
# streamlit header and interface
st.markdown(page_bg_img, unsafe_allow_html=True)
st.set_page_config(page_title='All Streams', layout='wide', page_icon=':cat:')
st.title("Azure Kinect Kit Stream")
st.write("This layout is made to see the live stream by Azure Kinect")
st.logo('assets/logo1.png')


main = st.container()
with main:    
    rgbContainer = st.container()
    otherContainer = st.container()
    bottomContainer = st.container()
    with rgbContainer:
        spaceForRGB: any = st.empty()
    with otherContainer:
        irCol, depthCol = st.columns(2)
        with irCol:
            spaceForIR: any = st.empty()
        with depthCol:
            spaceForDepth: any = st.empty()

with bottomContainer:
    option: any = st.selectbox(
    'Select modal?',
    ('Linear Regression', 'Object detector', 'People Counter'))
    st.write('You selected:', option)   
    
while True:
    # reading msgs sent by producer
    rgb_msg = rgConsumer.poll(timeout=1.0)
    depth_msg = depthConsumer.poll(timeout=1.0)
    ir_msg = irConsumer.poll(timeout=1.0)
    
    
    msgs = [rgb_msg, depth_msg, ir_msg]
    # checking for error 
    for msg in msgs:
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaException.KafkaError._PARTITION_EOF:
                continue
            else:
                print(msg.error())
                break

    color_image = rgb_msg.value()
    depth_msg = depth_msg.value()
    ir_msg = ir_msg.value()
    
    # converting bytes to images 
    color_image = bytes_to_image(color_image)
    depth_image = bytes_to_image(depth_msg)
    ir_image = bytes_to_image(ir_msg)
    
    # displaying images
    with rgbContainer:
        spaceForRGB.image(color_image,caption="RGB Stream",channels= "RGB")
    with irCol:
        spaceForIR.image(ir_image,caption="IR Stream",channels= "RGB", use_column_width= True)
    with depthCol:
        spaceForDepth.image(depth_image,caption="Depth Camera Stream",channels= "RGB", use_column_width= True)
    
