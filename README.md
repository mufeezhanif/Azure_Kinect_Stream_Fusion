# AzureKinectStreamFusion

## Overview
**AzureKinectStreamFusion** is a real-time data streaming and fusion project using the Azure Kinect SDK. The project processes RGB and infrared camera outputs, applying matrix addition and concatenation fusion techniques to produce synchronized multimodal visualizations. These are streamed live to a web interface built with Streamlit, allowing users to interact with and observe sensor fusion in action.

## Features
- **Real-time Data Streaming**: Captures live RGB and IR camera feeds from Azure Kinect.
- **Data Fusion**: Applies matrix addition and concatenation to combine RGB and IR data.
- **Interactive Web Interface**: Uses Streamlit for live visualization of the processed data.

## Installation

### Prerequisites
- **Python 3.8+**
- **Azure Kinect SDK**
- **Streamlit**
- **OpenCV**
- **numpy**

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/username/AzureKinectStreamFusion.git
   cd AzureKinectStreamFusion

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
3. Ensure the Azure Kinect SDK is correctly installed on your system. For more information, refer to <a href = "https://learn.microsoft.com/en-us/azure/kinect-dk/" target="_blank">Azure Kinect SDK Documentation</a>.
   
5. Run the application
   ```bash
   streamlit run app.py


## Conclusion
The **AzureKinectStreamFusion** project leverages the power of the Azure Kinect SDK and advanced data fusion techniques to deliver synchronized multimodal visualizations in real time. By combining RGB and infrared data streams with matrix addition and concatenation, this project showcases innovative sensor fusion in a user-friendly and interactive web interface using Streamlit.

Feel free to contribute or adapt this project for your own needs, and reach out if you have any questions!

