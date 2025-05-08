# Attendance System

## Our Team
- Ammar Yasser &emsp;&emsp;     **(Team member)**   &emsp;*AIU* 
- Jamal Khaled &emsp;&emsp;&ensp;     **(Team member)** &emsp;*AIU* 
- Seif Elkerdany &emsp;&emsp;&ensp;    **(Team leader)** &emsp;&ensp;&nbsp; *AIU* 
- Shrouq Waleed &emsp;&emsp;       **(Team member)**&emsp; *AIU* 

## Features

- **Universal Course Integration**: The system is built to support attendance tracking for any course offered at the university, providing a scalable and adaptable solution.
- **Seamless Student Enrollment**: Enrolling students is straightforward—just upload one or more reference images per student. The system will automatically use these for future attendance verification.
- **Camera-Based Face Recognition**: Attendance is recorded by simply scanning students' faces in real-time as they look at the camera, eliminating the need for manual input or cards.
- **Intelligent Face Matching**: The system uses advanced face verification techniques to compare live-captured images with the enrolled database and mark attendance accurately.

## Project Outline

- ### 1. Data collection & annotation
     -  ##### 1.1 Collecting Student faces.
        -  Different face reactions.
        -  Different light conditions.
        -  Every student must submit from 2 to 5 images.
    -   ##### 1.2 Explore the Data
        - Know more about the data.
        - The number of images.
        - The images formats.
        - The brightness of the images.
        - The sharpness and the quality of the images.
    -   ##### 1.3 Face detection and cropping
        - Using MTCNN to detect and crop faces.
    -   ##### 1.4 Data preprocessing
        - Resize images.
        - Remove noise from images.
        - Normalize the brightness.
    -   ##### 1.5 Generate pairs for training
        - Organize images in a directory.
        - Split the dataset into train, test and validation.
        - Create PyTorch dataset class.
- ### 2. Neural Network
    - ##### 2.1 Model architecture
        - Implement the [Siamese Neural Network](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) with modifications.
    - ##### 2.2 Data augmentation
        - Horizontal flip
        - Random rotation
        - Change brightness
        - Zoom out some images
        - Changing contrast
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
- ### 4. Deploying as a website with a database.

## Future updates
- **Edge Device Optimization**: Enhance the system’s efficiency to run seamlessly on edge devices such as Raspberry Pi and NVIDIA Jetson, enabling offline and portable deployment.
- **Real-Time Video Processing**: Extend functionality to support real-time video streams for continuous face detection and attendance logging, rather than relying solely on still images.
- **Accuracy Enhancement**: Improve the face verification model’s accuracy and robustness to meet high standards.
