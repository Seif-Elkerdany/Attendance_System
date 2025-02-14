import os
import cv2
import uuid

if __name__ == "__main__":
    # Taking the student id bec each image will be captured will have the name StudentID_ + uuid
    StudentID = input("Type your student ID: ")

    # Choosing and accessing the camera on the computer
    cap = cv2.VideoCapture(0)

    TRAINING_PATH = r"add here the path to training data folder"
    VERIFICATION_PATH = r"add here the path to verification data folder"

    # While the camera is working
    while cap.isOpened():
    
        # We take take the current frame
        ret, frame = cap.read()

        # Cut down frame to 178x218 pixels
        frame = frame[120:120+218, 200:200+178, :]

        # Collect images for training just press 't' to capture the image
        if cv2.waitKey(1) & 0XFF == ord('t'):
            imgname = os.path.join(TRAINING_PATH, '{}.jpg'.format(StudentID + uuid.uuid1())) # Creating the image path and name

            # Changing the images into grayscale 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Save it
            cv2.imwrite(imgname, frame)

        # Collect images for verification folder press 'v'
        if cv2.waitKey(1) & 0XFF == ord('v'):
            imgname = os.path.join(VERIFICATION_PATH, '{}.jpg'.format(StudentID + uuid.uuid1()))

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imwrite(imgname, frame)

        cv2.imshow('Image Collection', frame) # renaming the window
        
        # You can exit by pressing 'q'
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

    # just closing the window and the camera
    cap.release()
    cv2.destroyAllWindows()