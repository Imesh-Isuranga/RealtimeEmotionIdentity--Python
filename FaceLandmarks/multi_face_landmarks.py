import face_recognition
from PIL import Image,ImageDraw

#load the jpg file into a numpy array
face_image = face_recognition.load_image_file("Images/Testing/1.jpg")

#find all face landmarks and print the list to check
face_landmarks_list = face_recognition.face_landmarks(face_image)

#print all the face landmarks
print(len(face_landmarks_list))
#convert numpy array to pill image and create a Draw Object
pil_image = Image.fromarray(face_image)
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

#display the image
pil_image.show()