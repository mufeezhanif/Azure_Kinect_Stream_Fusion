import streamlit as st
import numpy as np
import cv2
from pyk4a import PyK4A, Config
import pyk4a
import base64

def imgAtBase64(file):
    with open(file,"rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def colorize_depth_image(depth_image)-> np.array:
    depth_image: np.array = np.clip(depth_image, 0, 5000) 
    depth_image: np.array = (depth_image / 5000 * 255).astype(np.uint8) 
    return cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

def colorize_ir_image(ir_image:np.array) -> np.array:
    # ir_image: np.array = np.clip(ir_image, 0, 500) 
    ir_image: np.array = (ir_image / 500 * 255).astype(np.uint8) 
    # return ir_image
    return cv2.applyColorMap(ir_image, cv2.COLORMAP_JET)

def matrix_addition_fusion(color_image: np.ndarray, ir_image:np.ndarray)-> np.ndarray:
    
    ir_resized_image = cv2.resize(ir_image, (color_image.shape[1],color_image.shape[0]))
    
    resized_ir_image_3channel = cv2.merge([ir_resized_image, ir_resized_image, ir_resized_image])

    fused_image = cv2.addWeighted(color_image, 0.5, resized_ir_image_3channel, 0.5, 0)

    return fused_image
    
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

st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("Azure Kinect Kit Stream")
st.write("This layout is made to see the live stream by Azure Kinect")
st.logo('assets/logo1.png')




# setting device configurations
device_config = Config(
    color_resolution=pyk4a.ColorResolution.RES_1080P,
    depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,
    color_format = pyk4a.ImageFormat.COLOR_BGRA32,
)

device = PyK4A(config=device_config)

# //Starting Device
device.start()
    
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
    capture = device.get_capture()
    
    color_image: np.array = capture.color[:, :, :3]  
    color_image: np.array = cv2.cvtColor(color_image, cv2.COLOR_BGRA2RGB)
    depth_image: np.array = colorize_depth_image(capture.depth)
    
    ir_image: np.array = colorize_ir_image(capture.ir)
    print(f'rgb rgb shape {capture.color.shape}')
    print(f'rgb depth shape {capture.depth.shape}')
    print(f'rgb ir shape {capture.ir.shape}')
    matrix_fused_fusion = matrix_addition_fusion(color_image=color_image,ir_image=ir_image)
    with rgbContainer:
        spaceForRGB.image(matrix_fused_fusion,caption="RGB Stream",channels= "RGB", use_column_width= True)
    with irCol:
        spaceForIR.image(ir_image,caption="IR Stream",channels= "RGB", use_column_width= True)
    with depthCol:
        spaceForDepth.image(color_image,caption="Concated Image",channels= "RGB", use_column_width= True)
    

device.stop()
device.close()
print('Done')
