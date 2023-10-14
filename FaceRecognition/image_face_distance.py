#importing require libraries
import face_recognition
import cv2


image_to_recognize_path = '../Images/1.jpg'

#loading the image to detect
original_image = cv2.imread(image_to_recognize_path)

#loading the sample images and get the 128 face embedddings from them
first_image = face_recognition.load_image_file('Images/Samples/1.jpg')
first_face_encoding = face_recognition.face_encodings(first_image)[0]

second_image = face_recognition.load_image_file('Images/Samples/2.jpg')
second_face_encoding = face_recognition.face_encodings(second_image)[0]


#create an array to save encodings
known_face_encodings = [first_face_encoding,second_face_encoding]

#create another array to hold labels
known_face_names = ["First Image","Second Image"]


#load the unknown image to recognize faces in it
image_to_recognize = face_recognition.load_image_file(image_to_recognize_path)
image_to_recognize_encodings = face_recognition.face_encodings(image_to_recognize)[0]


#find the face distance of image to recognize with the known samples
face_distances = face_recognition.face_distance(known_face_encodings, image_to_recognize_encodings)

#print face distance for each known sample to the unknown image
for i,face_distances in enumerate(face_distances):
    print("The calculate face distance is {:.2} from sample image {}".format(face_distances, known_face_names[i]))

