# Attendance System

## Our Team
- Ammar Yasser &emsp;&emsp;     **(Team member)**   &emsp;*AIU* 
- Jamal Khaled &emsp;&emsp;&ensp;     **(Team member)** &emsp;*AIU* 
- Seif Elkerdany &emsp;&emsp;&ensp;    **(Team leader)** &emsp;&ensp;&nbsp; *AIU* 
- Shrouq Waleed &emsp;&emsp;       **(Team member)**&emsp; *AIU* 

## Features

- This system is designed to take attendance for any course in the university.
- It is designed to just add 2 or more images for the students enrolled in this course and it will easily take attendance.
- You can use it by uploading image that contains the students attended the lecture or by scaning faces of students by looking at the camera.
- This project is done using Siamese Neural Network "SNN" to compare the currently captured image and the images in enrolled folder.

## Project Outline

- ### 1. Data collection & annotation
     -  ##### 1.1 Collecting Student faces.
        -  Different face reactions.
        -  Different light conditions.
        -  Every student must submit minimum 2 photos and maximum 5 images.
    -   ##### 1.2 Explore the Data
        - Know more about the data.
        - The number of images.
        - The brightness of the images.
        - The sharpness and the quality of the images.
    -   ##### 1.3 Face detection and cropping
        - Using MTCNN "TensorFlow".
    -   ##### 1.4 Data preprocessing
        - Resize images.
        - Use greyscale images.
        - Remove noise from images.
        - Normalize the brightness.
    -   ##### 1.5 Generate pairs for training
        - Organize images in a directory.
        - Split the dataset into train, test and validation.
        - Create PyTorch dataset class.
- ### 2. Siamese Neural Network
    - ##### 2.1 Model architecture
        - Implement the [Siamese Neural Network](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
    - ##### 2.2 Data augmentation
        - Horizontal flip
        - Random rotation
        - Change brightness
    - ##### 2.3 Training loop
        - Loss function
        - LR schedular
        - Early stopping and save the best weights
    - ##### 2.4 Evaluating the model
        - Keep updating and changing the model until it reaches great accuracy.
- ### 3. Attendance checking
    - ##### 3.1 Classroom image processing pipeline
    - ##### 3.2 One-to-many face matching
        - Create enrollment directory
    - ##### 3.3 Attendance logging
    - ##### 3.4 Save attendance records 
        - Save in CSV file format / xlsx file format
- ### 4. Deploying as a website

## Future updates
- Optimize the pipeline to work on Edge devices.
- Increase the model accuracy to integrate it with the university system.
