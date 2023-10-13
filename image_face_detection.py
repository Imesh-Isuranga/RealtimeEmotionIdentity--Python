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