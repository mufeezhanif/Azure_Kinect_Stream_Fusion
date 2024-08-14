import streamlit as st
import numpy as np
import cv2
from pyk4a import PyK4A, Config
import pyk4a
import base64

# encoding image
def imgAtBase64(file):
    with open(file,"rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# interface background images
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

# normalize images
def colorize_depth_image(depth_image)-> np.array:
    depth_image: np.array = np.clip(depth_image, 0, 5000) 
    depth_image: np.array = (depth_image / 5000 * 255).astype(np.uint8) 
    return cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

def colorize_ir_image(ir_image) -> np.array:
    ir_image: np.array = np.clip(ir_image, 0, 500) 
    ir_image: np.array = (ir_image / 500 * 255).astype(np.uint8) 
    return cv2.applyColorMap(ir_image, cv2.COLORMAP_JET)

# azure kinect sdk device configurations
device_config = Config(
    color_resolution=pyk4a.ColorResolution.RES_1080P,
    depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,
    color_format = pyk4a.ImageFormat.COLOR_BGRA32,
)

device = PyK4A(config=device_config)
device.start()

# streamlit interface
st.markdown(page_bg_img, unsafe_allow_html=True)
st.set_page_config(page_title='Multimodal Fusion', layout='wide')
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
    
# main functional loop
i = 0
while True:
    
    capture = device.get_capture()
    # to separate each color output
    color_image: np.array = capture.color[:, :, :3]   #to get only three channels i.e rgb
    color_image: np.array = cv2.cvtColor(color_image, cv2.COLOR_BGRA2RGB)
    depth_image: np.array = colorize_depth_image(capture.depth)
    ir_image: np.array = colorize_ir_image(capture.ir)

    # streaming the ouput on streamlit webpage
    with rgbContainer:
        spaceForRGB.image(color_image,caption="RGB Stream",channels= "RGB")
    with irCol:
        spaceForIR.image(ir_image,caption="IR Stream",channels= "RGB", use_column_width= True)
    with depthCol:
        spaceForDepth.image(depth_image,caption="Depth Camera Stream",channels= "RGB", use_column_width= True)
    i += 1

device.stop()
device.close()
print('Done')
