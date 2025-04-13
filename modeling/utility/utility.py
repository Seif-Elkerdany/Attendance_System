# modules
import os
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np

class datacleaner:
    """
    In this class I will put all the functions We will need for cleaning all the datasets.

    You can upload this file on kaggle and import from it what you need easily.
    """

    def extention_checker(folder_path):
        """
        This function takes the path of the folder contains the images and check how many images are .jpg and the non-jpg images
        also it returns the path of each non-jpg image.

        parameters:
            folder_path (String): the path of the folder.
        returns:
            list contains the path of non-jpg images.
        """

        # List to store non-JPG image paths
        non_jpg_images = []
        jpg_counter = 0     # counter to count the jpg images in the folder

        # Iterate through files in the folder
        for filename in tqdm(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):  # Check if it's a file
                ext = os.path.splitext(filename)[1].lower()  # Get file extension
                if ext != ".jpg":
                    non_jpg_images.append(file_path)  # Add to list if not JPG
                else:
                    jpg_counter += 1    # Add 1 to the jpg_counter if it is .jpg

        print("Number of JPG images found:", jpg_counter)
        print("Number of non-JPG images found:", len(non_jpg_images))

        return non_jpg_images

    def sample_plot(folder_path):
        """
        This function takes the path to an image folder and plots 9 random images from this folder.
        
        Parameters:
            folder_path (str): Path to the folder containing images.
        """
        
        # Get a list of image files in the folder
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))]

        if len(image_files) == 0:
            print("No images found in the folder.")
            return
    
        # Select 9 random images 
        sample_images = random.sample(image_files, min(9, len(image_files)))
    
        # Create a 3x3 subplot
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    
        for ax, img_file in zip(axes.flatten(), sample_images):
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)  # Read the image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB 
            ax.imshow(img)
            ax.set_title(img_file, fontsize=10)
            ax.axis('off')  # Hide axes
    
        plt.tight_layout()
        plt.show()

    def blur_detector(img_path):
        """
        This function calculates the variance of the Laplacian of the image to determine its sharpness.
    
        The Laplacian operator detects edges by computing the second derivative of the image. 
        OpenCV typically uses a 3x3 Laplacian kernel:
        
            [[ 1,  1,  1],
             [ 1, -8,  1],
             [ 1,  1,  1]]
        
        For more information, refer to the OpenCV documentation.
    
        Parameters:
            img_path (str): Path to the image file.
        
        Returns:
            float or None: The variance of the Laplacian, where lower values indicate more blur.
                           Returns None if the image cannot be loaded.
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        if img is None:
            return None  # Handle error without printing
    
        variance = cv2.Laplacian(img, cv2.CV_64F).var()  # Compute Laplacian variance
        return variance

    def check_blur_in_folder(folder_path):
        """
        This function calculates the blur score for all images in a folder,
        sorts them by sharpness, and plots the 5 most blurry and 5 sharpest images.
    
        Parameters:
            folder_path (str): Path to the folder containing images.

        Returns:
            tuple contains most_blurry, sharpest
        """
        blur_scores = {}  # Dictionary to store image names and their blur scores
        
        # Iterate over all files in the folder
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)  
            blur_score = datacleaner.blur_detector(img_path)  # Compute blur score using Laplacian variance
            
            if blur_score is not None: 
                blur_scores[file] = blur_score  # Store the blur score
    
        # Sort images by blur score (low values = blurry images)
        sorted_blur = sorted(blur_scores.items(), key=lambda x: x[1])  
    
        # Get the 5 most blurry and 5 sharpest images
        most_blurry = sorted_blur[:5]  # 5 images with the lowest blur score
        sharpest = sorted_blur[-5:]  # 5 images with the highest blur score
    
        # Plot the 5 most blurry images
        plt.figure(figsize=(12, 6))
        for i, (img_name, _) in enumerate(most_blurry):
            img_path = os.path.join(folder_path, img_name) 
            img = cv2.imread(img_path)  
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
            plt.subplot(2, 5, i + 1) 
            plt.imshow(img)
            plt.title(f"Blurry")  
            plt.axis("off") 
    
        # Plot the 5 sharpest images
        for i, (img_name, _) in enumerate(sharpest):
            img_path = os.path.join(folder_path, img_name)  
            img = cv2.imread(img_path)  
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
            
            plt.subplot(2, 5, i + 6)  
            plt.imshow(img)
            plt.title(f"Sharp") 
            plt.axis("off") 
    
        plt.suptitle("Blurry vs. Sharp Images", fontsize=16)  
        plt.tight_layout()  
        plt.show()  

        return most_blurry, sharpest

    def compute_brightness(img_path):
        """
        Calculates the mean brightness of an image using the HSV color space.
        
        Parameters:
            img_path (str): Path to the image file.
        
        Returns:
            float: The mean brightness value (0-255).
        """
        img = cv2.imread(img_path)  
        if img is None:
            return None  
    
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to HSV
        brightness = np.mean(hsv[:, :, 2])  # Extract V channel (brightness) and compute mean
        return brightness
    
    def check_brightness_in_folder(folder_path):
        """
        Computes brightness for all images in a folder and plots the distribution.
    
        Parameters:
            folder_path (str): Path to the folder containing images.

        Returns:
            sorted_brightness (list): List contains all sorted images
        """
        brightness_scores = {}  # Store brightness values
    
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            brightness = datacleaner.compute_brightness(img_path)
            
            if brightness is not None:
                brightness_scores[file] = brightness
    
        # Sort images by brightness
        sorted_brightness = sorted(brightness_scores.items(), key=lambda x: x[1])
    
        # Plot brightness distribution
        plt.figure(figsize=(8, 5))
        plt.hist(list(brightness_scores.values()), bins=30, color='blue', alpha=0.7)
        plt.axvline(x=np.mean(list(brightness_scores.values())), color='red', linestyle='dashed', label='Mean Brightness')
        plt.xlabel("Brightness Level (0-255)")
        plt.ylabel("Number of Images")
        plt.title("Brightness Distribution of Images")
        plt.legend()
        plt.show()

        return sorted_brightness
    
    def normalize_brightness(image_path):
        """
        Normalizes the brightness of an image using histogram equalization.
        Converts the image to grayscale after normalization.

        Args:
            image_path (str): Path to the image.

        Returns:
            PIL Image: Brightness-normalized grayscale image.
        """
        img = cv2.imread(image_path) 
        if img is None:
            print(f"Skipping: {image_path} (Invalid image file)")
            return None

        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # Convert to LAB color space
        l, a, b = cv2.split(img_lab)  # Split LAB channels

        # Apply histogram equalization to the L (lightness) channel
        l_eq = cv2.equalizeHist(l)
        img_lab_eq = cv2.merge((l_eq, a, b))

        # Convert back to BGR color space
        img_eq = cv2.cvtColor(img_lab_eq, cv2.COLOR_LAB2BGR)

        # Convert to grayscale
        img_gray = cv2.cvtColor(img_eq, cv2.COLOR_BGR2GRAY)

        return Image.fromarray(img_gray)  # Convert to PIL Image
    
    def process_and_overwrite_images(image_folder):
        """
        Processes all images in a folder:
        - Normalizes brightness
        - Converts to grayscale
        - Overwrites original image files

        Args:
            image_folder (str): Path to the folder containing images.
        """
        image_list = [img for img in os.listdir(image_folder) if img.lower().endswith(".jpg")]

        if not image_list:
            print("Error: No valid images found!")
            return

        for img_name in tqdm(image_list):
            img_path = os.path.join(image_folder, img_name)

            # Process image
            img_processed = datacleaner.normalize_brightness(img_path)
            if img_processed is not None:
                # Overwrite the original image
                img_processed.save(img_path)