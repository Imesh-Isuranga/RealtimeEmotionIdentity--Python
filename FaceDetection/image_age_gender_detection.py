#importing require libraries
import face_recognition
import cv2

#Get the webcam #0 (the default one,1,2 etc means additional attached cams)

image_to_detect = cv2.imread('Images/1.jpg')

#Initalize empty array for each face locations
all_faces_locations = face_recognition.face_locations(image_to_detect,model="hog")


#looping through the face locations
for index,current_face_location in enumerate(all_faces_locations):
    #splitting the tuple to et the four position values
    top_pos,right_pos,bottom_pos,left_pos = current_face_location
    #printing the location of current face
    print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1, top_pos,right_pos,bottom_pos,left_pos))
    
    #paste the blurred face into the actual frame
    current_face_image = image_to_detect[top_pos:bottom_pos,left_pos:right_pos] 
    
    
    #The 'AGE_GENDER_MODEL_MEAN_VALUES' calculated by using the numpy.mean()
    AGE_GENDER_MODEL_MEAN_VALUES = (78.4263377603,87.7689143744,114.895847746)
    
    #create blob of current flace slice
    current_face_image_blob = cv2.dnn.blobFromImage(current_face_image,1,(227,227),AGE_GENDER_MODEL_MEAN_VALUES,swapRB=False)
    
    gender_label_list = ['Male','Female']
    gender_protext = "dataset/gender_deploy.prototxt"
    gender_caffemodel = "dataset/gender_net.caffemodel"
    gender_cov_net = cv2.dnn.readNet(gender_caffemodel, gender_protext)
    gender_cov_net.setInput(current_face_image_blob)
    gender_prediction = gender_cov_net.forward()
    gender = gender_label_list[gender_prediction[0].argmax()]
    
    #repeat same steps for age prediction
    age_label_list = ['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']
    age_protext = "dataset/age_deploy.prototxt"
    age_caffemodel = "dataset/age_net.caffemodel"
    age_cov_net = cv2.dnn.readNet(age_caffemodel, age_protext)
    age_cov_net.setInput(current_face_image_blob)
    age_prediction = age_cov_net.forward()
    age = age_label_list[age_prediction[0].argmax()]
    
    #print gender and age under the rectangle
    #draw rectangle around each face location in the main video frame insode the loop
    cv2.rectangle(image_to_detect, (left_pos,top_pos), (right_pos,bottom_pos),(0,0,255),2)
    #display the name as text in the image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image_to_detect, gender + " " + age + "yrs" , (left_pos,bottom_pos),font,1, (0,255,0),1)    
        
    #showing the current face with rectangle drawn
    cv2.imshow("Age & Gender",image_to_detect)
