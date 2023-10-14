#importing require libraries
import face_recognition
import cv2


#Get the webcam #0 (the default one,1,2 etc means additional attached cams)
webcam_video_stream = cv2.VideoCapture('Videos/example.mp4')


#loading the sample images and get the 128 face embedddings from them
first_image = face_recognition.load_image_file('Images/Samples/1.jpg')
first_face_encoding = face_recognition.face_encodings(first_image)[0]

second_image = face_recognition.load_image_file('Images/Samples/3.jpg')
second_face_encoding = face_recognition.face_encodings(second_image)[0]


#create an array to save encodings
known_face_encodings = [first_face_encoding,second_face_encoding]

#create another array to hold labels
known_face_names = ["First Image","Second Image"]





all_faces_locations = []
all_face_encodings = []
all_face_names = []




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
    #find all the faces and face encodings in the unknown image
    all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=1,model="hog")
    all_face_encodings = face_recognition.face_encodings(current_frame_small,all_face_locations)
    
    #loop through each face location and face encodings found in the unknown image
    for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
        #split the tuple
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4
        
        #compare faces and get the matches list(inside the loop)
        #See if the face is any match(es) for the known face(s) 
        all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
        
        #initialize name string(inside the loop)
        name = "Unknown Face"
        
        #If a match was found in known_face_encodings,use the first one
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name = known_face_names[first_match_index]
        
        #draw blue rectangle around the face
        cv2.rectangle(current_frame, (left_pos,top_pos), (right_pos,bottom_pos),(255,0,0),2)
        
        #write name below face(inside loop)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)
        
        #showing the current face with rectangle drawn
        cv2.imshow("Webcam Video",current_frame)
        
        #Press 'q' on the keyboard to break the while loop!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    #release the webcam resource
webcam_video_stream.release()
cv2.destroyAllWindows()

      
        
