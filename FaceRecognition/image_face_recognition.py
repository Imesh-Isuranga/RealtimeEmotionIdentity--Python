#importing require libraries
import face_recognition
import cv2
import os

#loading the image to detect
original_image = cv2.imread('../Images/1.jpg')

#loading the sample images and get the 128 face embedddings from them
first_image = face_recognition.load_image_file('Images/Samples/1.jpg')
first_face_encoding = face_recognition.face_encodings(first_image)[0]

second_image = face_recognition.load_image_file('Images/Samples/2.jpg')
second_face_encoding = face_recognition.face_encodings(second_image)[0]


#create an array to save encodings
known_face_encodings = [first_face_encoding,second_face_encoding]

#create another array to hold labels
known_face_names = ["First Image","Second Image"]

#load an unknown image to identify the faces
image_to_recognize = face_recognition.load_image_file("Images/Testing/1.jpg")

#find all the faces and face encodings in the unknown image
all_face_locations = face_recognition.face_locations(image_to_recognize,model="hog")
all_face_encodings = face_recognition.face_encodings(image_to_recognize,all_face_locations)


#loop through each face location and face encodings found in the unknown image
for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
    #split the tuple
    top_pos,right_pos,bottom_pos,left_pos = current_face_location
    
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
    cv2.rectangle(original_image, (left_pos,top_pos), (right_pos,bottom_pos),(255,0,0),2)
    
    #write name below face(inside loop)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(original_image, name, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)
    
    #show the image with rectangle and text(inside the loop)
    cv2.imshow('Identified Faces',original_image)
  
    
