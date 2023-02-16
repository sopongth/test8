import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2)
font = cv2.FONT_HERSHEY_SIMPLEX

st.title("หันซ้าย หันขวา")

class VideoProcessor:  
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        #--------------------------------------------------
        with mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5) as face_mesh: 
            img = cv2.flip(img,1) 
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img.flags.writeable = False
            results = face_mesh.process(img)
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
              for face_landmarks in results.multi_face_landmarks:
                h, w, c = img.shape
                point137 = face_landmarks.landmark[137]
                point137_x = int(point137.x *w)
                point137_y = int(point137.y *h)
                cv2.circle(img,(point137_x,point137_y),8,(0,0,255),-1)
                
                point4 = face_landmarks.landmark[4]
                point4_x = int(point4.x *w)
                point4_y = int(point4.y *h)
                cv2.circle(img,(point4_x,point4_y),8,(0,255,0),-1)
                
                point366 = face_landmarks.landmark[366]
                point366_x = int(point366.x *w)
                point366_y = int(point366.y *h)
                cv2.circle(img,(point366_x,point366_y),8,(255,0,0),-1)

                cv2.line(img,(point4_x,point4_y),(point137_x,point137_y),(0,0,255),5)
                cv2.line(img,(point4_x,point4_y),(point366_x,point366_y),(255,0,0),5)

                distL = int(point4_x-point137_x)
                distR = int(point366_x-point4_x)            
                 
                if (distL > distR) and (distL - distR) > 40 :
                    cv2.putText(img,"Right",(50,100),font,2,(0,255,0),8)
                elif (distR > distL) and (distR - distL) > 40 :
                    cv2.putText(img,"Left",(50,100),font,2,(0,100,255),8)
                else:
                    cv2.putText(img,"Straight",(50,100),font,2,(0,255,255),8)
        #--------------------------------------------------
        return av.VideoFrame.from_ndarray(img,format="bgr24")

webrtc_streamer(key="test",
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True,"audio": False},
                rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]})

