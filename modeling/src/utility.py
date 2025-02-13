# modules
import os
from tqdm import tqdm


class datacleaner:
    """
    In this class I will put all the functions We will need for cleaning all the datasets.

    You can upload this file on kaggle and import from it what you need easily.

    The functions are:
        1) extention_checker
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

        # Print the result
        print("Number of JPG images found:", jpg_counter)
        print("Number of non-JPG images found:", len(non_jpg_images))

        return non_jpg_images