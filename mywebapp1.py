import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np

st.title("ตรวจจับวัตถุสีแดง")

class VideoProcessor:
    def __init__(self):
        self.th = 150
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        #---------------------------------------------
        img = cv2.flip(img,1) 
        imgYCrCb  = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(imgYCrCb)
        Cr = channels[1]
        ret,BW = cv2.threshold(Cr,self.th,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(BW,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt = contours[max_index]
            x,y,w,h = cv2.boundingRect(cnt)    
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),4) 
        #---------------------------------------------
        return av.VideoFrame.from_ndarray(img,format="bgr24")

ctx = webrtc_streamer(key="test",
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True,"audio": False})

if ctx.video_processor:
    ctx.video_processor.th = st.slider("เลือกระดับความเข้มของสี",0,255,150)
