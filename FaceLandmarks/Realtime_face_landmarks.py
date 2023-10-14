import face_recognition
from PIL import Image,ImageDraw
import cv2
import numpy as np


#Get the webcam #0 (the default one,1,2 etc means additional attached cams)
webcam_video_stream = cv2.VideoCapture(0)


#Initalize empty array for each face locations
all_faces_locations = []

#create an outer while loop to loop through each frame of video
while True:   
    #get single frame of video as image
    #get current frame
    ret,current_frame = webcam_video_stream.read()
    
    
    #find all face landmarks and print the list to check
    face_landmarks_list = face_recognition.face_landmarks(current_frame)
    
    #print all the face landmarks
    print(len(face_landmarks_list))
    #convert numpy array to pill image and create a Draw Object
    pil_image = Image.fromarray(current_frame)
    #convert the pil image to draw object
    d = ImageDraw.Draw(pil_image)


    #loop through every face
    index=0
    while index<len(face_landmarks_list):
        #for loop to iterate through all face landmarks
        for face_landmarks in face_landmarks_list:
            
            d.line(face_landmarks['chin'],fill=(255,255,255),width=2)
            d.line(face_landmarks['left_eyebrow'],fill=(255,255,255),width=2)
            d.line(face_landmarks['right_eyebrow'],fill=(255,255,255),width=2)
            d.line(face_landmarks['nose_bridge'],fill=(255,255,255),width=2)
            d.line(face_landmarks['nose_tip'],fill=(255,255,255),width=2)
            d.line(face_landmarks['left_eye'],fill=(255,255,255),width=2)
            d.line(face_landmarks['right_eye'],fill=(255,255,255),width=2)
            d.line(face_landmarks['top_lip'],fill=(255,255,255),width=2)
            d.line(face_landmarks['bottom_lip'],fill=(255,255,255),width=2)
            
            
        index += 1
        
    #convert PIL image to RGB to show in opencv window
    rgb_image = pil_image.convert('RGB')
    rgb_open_cv_image = np.array(pil_image)
    
    #convert RGB to BGR
    bgr_open_cv_image = cv2.cvtColor(rgb_open_cv_image, cv2.COLOR_RGB2BGR)
    bgr_open_cv_image = bgr_open_cv_image[:,:,::-1].copy()
    
    #showing the current face with rectangle drawn
    cv2.imshow("Webcam Video",bgr_open_cv_image)
        
    #Press 'q' on the keyboard to break the while loop!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#release the webcam resource
webcam_video_stream.release()
cv2.destroyAllWindows()
    
