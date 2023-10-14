#importing require libraries
import face_recognition
import cv2

#loading the image to detect
image_to_detect = cv2.imread('Images/1.jpg')

cv2.imshow("test", image_to_detect)


#find all faces locations using face_locations() function
#model can be `cnn` or `hog`
#number_of_times_to_upsample=1 higher and detect more faces

all_faces_locations = face_recognition.face_locations(image_to_detect,model="hog")

#printing the number of faces in the array
print("There are {} face(s) in this images".format(len(all_faces_locations)))

#looping through the face locations
for index,current_face_location in enumerate(all_faces_locations):
    #splitting the tuple to et the four position values
    top_pos,right_pos,bottom_pos,left_pos = current_face_location
    print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1, top_pos,right_pos,bottom_pos,left_pos))
    
    
    #slice image array by positions inside the loop
    current_face_image = image_to_detect[top_pos:bottom_pos,left_pos:right_pos]
    
    #show each sliced face inside the loop
    cv2.imshow("Face No: " + str(index+1),current_face_image)