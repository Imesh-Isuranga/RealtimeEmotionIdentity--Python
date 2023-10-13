#importing require libraries
import face_recognition
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json



#Get the webcam #0 (the default one,1,2 etc means additional attached cams)
webcam_video_stream = cv2.VideoCapture(0)

#face expression model initialization
face_exp_model = model_from_json(open("dataset/facial_expression_model_structure.json","r").read());
#load weights into model
face_exp_model.load_weights('dataset/facial_expression_model_weights.h5')
#list of emotions labels
emotions_label = ('angry','disgust','fear','happy','sad','surprice','neutral')

#Initalize empty array for each face locations
all_faces_locations = []

#create an outer while loop to loop through each frame of video
while True:   
    #get single frame of video as image
    #get current frame
    ret,current_frame = webcam_video_stream.read()
    
    
    #resize the frame to a quarter of size so that the computer can process it faster
    current_frame_small = cv2.resize(current_frame, (0,0),fx=0.25,fy=0.25)
    
    
    #find all faces locations using face_locations() function
    #model can be `cnn` or `hog`
    #number_of_times_to_upsample=1 higher and detect more faces
    
    all_faces_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=2,model="hog")

    
    #looping through the face locations
    for index,current_face_location in enumerate(all_faces_locations):
        #splitting the tuple to et the four position values
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4
        #printing the location of current face
        print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1, top_pos,right_pos,bottom_pos,left_pos))
        
        #slicing the current face from main image
        current_face_image = current_frame[top_pos:bottom_pos,left_pos:right_pos]
        
        
        #draw rectangle around each face location in the main video frame insode the loop
        cv2.rectangle(current_frame, (left_pos,top_pos), (right_pos,bottom_pos),(0,0,255),2)
        
        #preprocess input,convert it to an image like as the data in dataset 
        #convert to grayscale
        current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)
        #resize to 48x48 px size
        current_face_image = cv2.resize(current_face_image, (48,48))
        #convert the PIL image into a 3d numpy array
        img_pixels = image.img_to_array(current_face_image)
        #expand the shape of an array into single row multiple columns
        img_pixels = np.expand_dims(img_pixels,axis=0)
        #pixels are in range of[0,255].normalize all pixels in scale of [0,1]
        img_pixels /= 255
        
        #do prediction using model,get the prediction values for all 7 expressions
        exp_predictions = face_exp_model.predict(img_pixels)
        #find max indexed prediction value (0 till 7)
        max_index = np.argmax(exp_predictions[0])
        #get corresponding lable from emotions_label
        emotion_label = emotions_label[max_index]
        
        #display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, emotion_label, (left_pos,bottom_pos), font,1, (255,255,255),1)
        
    #showing the current face with rectangle drawn
    cv2.imshow("Webcam Video",current_frame)
        
    #Press 'q' on the keyboard to break the while loop!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#release the webcam resource
webcam_video_stream.release()
cv2.destroyAllWindows()
