import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import os
import matplotlib.pyplot as plt
import io
import random
from PIL import ImageEnhance
from mtcnn import MTCNN
import cv2
from google.colab import files
class dataCleaner_38K():
    detector = MTCNN() # save computation rather than initialization on every fun. call
    
    @staticmethod
    def loadDF():
        """
        returns pandas df of 38k instance with 3 col[image1,image2,target]
        """
        splits = {
        'agedb_30': 'data/agedb_30-00000-of-00001-a951a9443360b4cd.parquet',
        'calfw': 'data/calfw-00000-of-00001-494813e56bc84049.parquet',
        'cfp_ff': 'data/cfp_ff-00000-of-00001-699a051e1c50724b.parquet',
        'cfp_fp': 'data/cfp_fp-00000-of-00001-b24730458d082e8b.parquet',
        'cplfw': 'data/cplfw-00000-of-00001-f1a0092b98fb0677.parquet',
        'lfw': 'data/lfw-00000-of-00001-eedd8244e78ffd55.parquet'}
        
        dfs = []

        # Iterate over the splits and load each parquet file into a dataframe
        for split_name, file_path in splits.items():
          try:
            df = pd.read_parquet("hf://datasets/cat-claws/face-verification/" + file_path)
            dfs.append(df)
          except Exception as e:
              print(f"Error loading {split_name}: {e}")
        # Concatenate all the dataframes into a single dataframe
        if dfs:
          df_all = pd.concat(dfs, ignore_index=True)
        else:
          print("No dataframes were successfully loaded.")
        return df_all
    
    @staticmethod
    def check_image_bytes(image_bytes):
        """
        Check if an image in bytes format is valid and not corrupted.
    
        Parameters:
            image_bytes (bytes): Image data in bytes format.
    
        Returns:
            bool: True if the image is valid, False otherwise.
        """
        try:
            img = Image.open(BytesIO(image_bytes))
            img.verify()  # Verify that the file is an image
            return True
        except Exception:
            return False
    
# =============================================================================
#     WE WILL CONTINUE WORKING WITH THEM IN BYTES FORM UNTIL PREPROCESSING.
#     def save_bytes_as_jpg(image_bytes, output_path):
#         """
#         Convert image bytes to a JPG file and save it to disk.
#     
#         Parameters:
#             image_bytes (bytes): Image data in bytes format.
#             output_path (str): Path to save the JPG file.
#     
#         Returns:
#             None
#         """
#         try:
#             # Convert bytes to an image object
#             img = Image.open(BytesIO(image_bytes))
#     
#             # Convert to RGB mode (JPG doesn't support transparency)
#             if img.mode in ("RGBA", "P"):
#                 img = img.convert("RGB")
#     
#             # Save as JPG
#             img.save(output_path, "JPEG")
#             print(f"Saved image to {output_path}")
#         except Exception as e:
#             print(f"Error saving image: {e}")
# =============================================================================
    @staticmethod
    def remove_corrupted_images_bytes(df):
        """
        Remove rows where either `image1` or `image2` is corrupted and return a cleaned Pandas DataFrame.
    
        Parameters:
            df (pd.DataFrame): A DataFrame where each row contains `image1` and `image2` 
            as dictionaries with a "bytes" key.
    
        Returns:
            pd.DataFrame: A cleaned DataFrame with corrupted rows removed.
        """
        # Create a mask to identify valid rows
        mask = df.apply(lambda row: dataCleaner_38K.check_image_bytes(row["image1"]["bytes"]) and dataCleaner_38K.check_image_bytes(row["image2"]["bytes"]), axis=1)
    
        # Filter the DataFrame using the mask
        cleaned_df = df[mask]
    
        # Print the number of removed rows
        num_removed = len(df) - len(cleaned_df)
        print(f"Removed {num_removed} corrupted rows.")
    
        return cleaned_df
    @staticmethod
    def visualize_sample(df):
        if len(df) == 0:
            print("No images to visualize.")
            return
    
        random_indices = random.sample(range(len(df)), 3)

        fig, axes = plt.subplots(3, 2, figsize=(6, 6))
        
        for i, index in enumerate(random_indices):
            try:
                # Image 1
                image_data1 = df['image1'].iloc[index]['bytes']
                image1 = Image.open(io.BytesIO(image_data1))
                axes[i, 0].imshow(image1)
                axes[i, 0].axis('off')
                axes[i, 0].set_title(f"Image1 - Index: {index}")
                
                # Image 2
                image_data2 = df['image2'].iloc[index]['bytes']
                image2 = Image.open(io.BytesIO(image_data2))
                axes[i, 1].imshow(image2)
                axes[i, 1].axis('off')
                axes[i, 1].set_title(f"Image2 - Index: {index}, Target: {df['target'].iloc[index]}")
        
            except Exception as e:
                print(f"Error loading image at index {index}: {e}")
        
        plt.tight_layout()
        plt.show()
    @staticmethod  
    def check_image_mode(image_bytes):
        """
        Check if an image is grayscale or RGB.
    
        Parameters:
            image_bytes (bytes): Image data in bytes format.
    
        Returns:
            str: The image mode (e.g., "L" for grayscale, "RGB" for color).
        """
        try:
            img = Image.open(BytesIO(image_bytes))
            return img.mode
        except Exception as e:
            print(f"Error checking image mode: {e}")
            return None
    @staticmethod
    def convert_to_rgb(image_bytes):
        """
        Convert an image to RGB mode if it is grayscale.
    
        Parameters:
            image_bytes (bytes): Image data in bytes format.
    
        Returns:
            bytes: Image data in RGB mode.
        """
        try:
            img = Image.open(BytesIO(image_bytes))
            if img.mode == "L":  # Grayscale
                img = img.convert("RGB")  # Convert to RGB
            # Convert the image back to bytes
            byte_arr = BytesIO()
            img.save(byte_arr, format="JPEG")
            return byte_arr.getvalue()
        except Exception as e:
            print(f"Error converting image: {e}")
            return None
    @staticmethod
    def process_gray_rgb(df):
        """
        Process a dataset to ensure all images are in RGB mode.
    
        Parameters:
            df (pd.DataFrame): A DataFrame containing `image1` and `image2` as dictionaries with a "bytes" key.
    
        Returns:
            pd.DataFrame: A cleaned DataFrame with all images in RGB mode.
        """
        grayCount = 0
        for idx, row in df.iterrows():
            # Process image1
            if dataCleaner_38K.check_image_mode(row["image1"]["bytes"]) == "L":  # Grayscale
                df.at[idx, "image1"]["bytes"] = dataCleaner_38K.convert_to_rgb(row["image1"]["bytes"])
                grayCount += 1
            # Process image2
            if dataCleaner_38K.check_image_mode(row["image2"]["bytes"]) == "L":  # Grayscale
                df.at[idx, "image2"]["bytes"] = dataCleaner_38K.convert_to_rgb(row["image2"]["bytes"])
                grayCount += 1
        print(f"{grayCount} img out of {38000*2} were gray scale.")
        return df
    
    @staticmethod
    def calculate_brightness(image_bytes):
        """
        Calculate the brightness of an image as the average pixel intensity.
        
        Parameters:
            image_bytes (bytes): Image data in bytes format.
        
        Returns:
            float: Average brightness value.
        """
        try:
            img = Image.open(BytesIO(image_bytes)).convert("L")  # Convert to grayscale
            return np.mean(np.array(img))
        except Exception as e:
            print(f"Error calculating brightness: {e}")
            return None
    @staticmethod
    def plot_brightness_outliers(df, threshold_low=50, threshold_high=200):
        """
        Plot images with brightness values outside the specified thresholds.
        """
        # Calculate brightness for both columns
        brightness_image1 = df["image1"].apply(lambda x: dataCleaner_38K.calculate_brightness(x["bytes"]))
        brightness_image2 = df["image2"].apply(lambda x: dataCleaner_38K.calculate_brightness(x["bytes"]))
        
        # Get indices of outliers for both columns
        outlier_indices_image1 = df[(brightness_image1 < threshold_low) | (brightness_image1 > threshold_high)].index
        outlier_indices_image2 = df[(brightness_image2 < threshold_low) | (brightness_image2 > threshold_high)].index
        
        # Combine and deduplicate indices
        outlier_indices = outlier_indices_image1.union(outlier_indices_image2)
        outliers = df.loc[outlier_indices]  # Get the actual rows
        
        print(f"Found {len(outliers)} brightness outliers.")
        dataCleaner_38K.visualize_sample(outliers)
        
    @staticmethod
    def calculate_contrast(image_bytes):
        """
        Calculate the contrast of an image as the standard deviation of pixel intensities.
        
        Parameters:
            image_bytes (bytes): Image data in bytes format.
        
        Returns:
            float: Contrast value.
        """
        try:
            img = Image.open(BytesIO(image_bytes)).convert("L")  # Convert to grayscale
            return np.std(np.array(img))
        except Exception as e:
            print(f"Error calculating contrast: {e}")
            return None
    @staticmethod
    def plot_contrast_outliers(df, threshold_low=20, threshold_high=100):
        """
        Plot images with contrast values outside the specified thresholds for both `image1` and `image2`.
        """
        # Calculate contrast for both columns
        contrast_image1 = df["image1"].apply(lambda x: dataCleaner_38K.calculate_contrast(x["bytes"]))
        contrast_image2 = df["image2"].apply(lambda x: dataCleaner_38K.calculate_contrast(x["bytes"]))
        
        # Get indices of outliers for both columns
        outlier_indices_image1 = df[(contrast_image1 < threshold_low) | (contrast_image1 > threshold_high)].index
        outlier_indices_image2 = df[(contrast_image2 < threshold_low) | (contrast_image2 > threshold_high)].index
        
        # Combine and deduplicate indices
        outlier_indices = outlier_indices_image1.union(outlier_indices_image2)
        outliers = df.loc[outlier_indices]
        
        print(f"Found {len(outliers)} contrast outliers.")
        dataCleaner_38K.visualize_sample(outliers)

    @staticmethod
    def calculate_sharpness(image_bytes):
        """
        Calculate the sharpness of an image using the variance of the Laplacian.
        
        Parameters:
            image_bytes (bytes): Image data in bytes format.
        
        Returns:
            float: Sharpness value.
        """
        try:
            img = Image.open(BytesIO(image_bytes))
            img_np = np.array(img)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except Exception as e:
            print(f"Error calculating sharpness: {e}")
            return None
    @staticmethod
    def plot_sharpness_outliers(df, threshold=70):
        """
        Plot images with sharpness values below the specified threshold for both `image1` and `image2`.
        
        Returns:
        pd.DataFrame: DataFrame containing only the outlier rows.
        """
        # Calculate sharpness for both columns
        sharpness_image1 = df["image1"].apply(lambda x: dataCleaner_38K.calculate_sharpness(x["bytes"]))
        sharpness_image2 = df["image2"].apply(lambda x: dataCleaner_38K.calculate_sharpness(x["bytes"]))
        
        # Get indices of outliers for both columns
        outlier_indices_image1 = df[sharpness_image1 < threshold].index
        outlier_indices_image2 = df[sharpness_image2 < threshold].index
        
        # Combine and deduplicate indices
        outlier_indices = outlier_indices_image1.union(outlier_indices_image2)
        outliers = df.loc[outlier_indices]
        
        print(f"Found {len(outliers)} sharpness outliers.")
        dataCleaner_38K.visualize_sample(outliers)
        return outliers
    @staticmethod
    def remove_outliers(df, outliers):
        """
        Remove outlier rows from the DataFrame.
        
        Parameters:
            df (pd.DataFrame): Original DataFrame.
            outliers (pd.DataFrame): DataFrame of outliers to remove.
        
        Returns:
            pd.DataFrame: Cleaned DataFrame with outliers removed.
        """
        # Drop rows by index
        cleaned_df = df.drop(outliers.index)
        print(f"Removed {len(outliers)} outliers. Remaining rows: {len(cleaned_df)}")
        return cleaned_df
    @staticmethod
    def plot_low_resolution_outliers(df, min_resolution=(64, 64)):
        """
        Plot images that do not meet the minimum resolution requirement for either image1 or image2.
        """
        # Check resolution for both columns
        resolution_mask_image1 = df["image1"].apply(
            lambda x: dataCleaner_38K.check_resolution(x["bytes"], min_resolution)
        )
        resolution_mask_image2 = df["image2"].apply(
            lambda x: dataCleaner_38K.check_resolution(x["bytes"], min_resolution)
        )
        
        # Identify rows where either image1 or image2 fails
        outliers = df[~(resolution_mask_image1 & resolution_mask_image2)]
        
        print(f"Found {len(outliers)} low-resolution outliers.")
        dataCleaner_38K.visualize_sample(outliers)
    @staticmethod   
    def gamma_correction(img, gamma):
        """
        Apply gamma correction to an image with proper mode handling.
        it offers nonlinear transformation which will work better than the enhancer in this case.
        """
        # Convert to RGB/L mode first to ensure valid LUT application
        if img.mode not in ('L', 'RGB'):
            img = img.convert('RGB')
        
        # Build lookup table for 256 values
        table = [int(((i / 255.0) ** (1 / gamma)) * 255) for i in range(256)]
        
        # Apply to all channels if RGB
        if img.mode == 'RGB':
            table = table * 3  # Apply same correction to all 3 channels
        
        return img.point(table)
    
    @staticmethod
    def adjust_brightness(image_bytes, **kwargs):
        """
        Adjust brightness of an image based on defined thresholds.
        
        Brightness Thresholds:
          - Extreme Underexposure (<30): Remove image (return None)
          - Moderate Underexposure (30-50): Normalize using gamma correction (gamma=2.0)
          - Normal Range (50-200): Keep as is
          - Moderate Overexposure (200-220): Normalize using gamma correction (gamma=0.7)
          - Extreme Overexposure (>220): Remove image (return None)
        
        Parameters:
            image_bytes (bytes): Image data in bytes format.
            **kwargs: Additional arguments (optional).
        
        Returns:
            bytes: Adjusted image data in bytes format, or None if outlier.
        """
        try:
            brightness = dataCleaner_38K.calculate_brightness(image_bytes)
            if brightness is None:
                return None
    
            # Remove extremes
            if brightness < 30 or brightness > 220:
                return None
    
            img = Image.open(BytesIO(image_bytes)).convert('RGB')  # Force RGB mode
            
            if 30 <= brightness < 50:
                img = dataCleaner_38K.gamma_correction(img, 2.0)
            elif 200 < brightness <= 220:
                img = dataCleaner_38K.gamma_correction(img, 0.7)
    
            # Save with quality preservation
            byte_arr = BytesIO()
            img.save(byte_arr, format="JPEG", quality=95, subsampling=0)
            return byte_arr.getvalue()
            
        except Exception as e:
            print(f"Error adjusting brightness: {e}")
            return None

    @staticmethod
    def adjust_contrast(image_bytes, **kwargs):
        """
        Adjust contrast of an image based on defined thresholds.
        
        Contrast Thresholds:
          - Very Low Contrast (<15): Remove image (return None)
          - Low Contrast (15-25): Enhance contrast (using factor ~1.75)
          - Normal Contrast (25-100): Keep as is
          - High Contrast (>100): Reduce contrast (using factor ~0.8)
        
        Parameters:
            image_bytes (bytes): Image data in bytes format.
            **kwargs: Additional arguments (optional).
        
        Returns:
            bytes: Adjusted image data in bytes format, or None if outlier.
        """
        try:
            contrast = dataCleaner_38K.calculate_contrast(image_bytes)
            if contrast is None:
                return None

            # Remove very low contrast images
            if contrast < 15:
                return None

            img = Image.open(BytesIO(image_bytes))
            if 15 <= contrast < 25:
                contrast_factor = 1.75  # Enhance contrast
            elif 25 <= contrast <= 100:
                contrast_factor = 1.0   # Keep as is
            elif contrast > 100:
                contrast_factor = 0.8   # Reduce contrast
            
            enhancer = ImageEnhance.Contrast(img)
            adjusted_img = enhancer.enhance(contrast_factor)
            
            byte_arr = BytesIO()
            adjusted_img.save(byte_arr, format="JPEG")
            return byte_arr.getvalue()
        except Exception as e:
            print(f"Error adjusting contrast: {e}")
            return None

    @staticmethod
    def adjust_sharpness(image_bytes, **kwargs):
        """
        Adjust sharpness of an image based on defined thresholds.
        
        Sharpness Thresholds (using Laplacian variance):
          - Severe Blur (<50): Remove image (return None)
          - Moderate Blur (50-70): Apply sharpening filter (factor 2.0)
          - Acceptable Sharpness (>=70): Keep as is
        
        Parameters:
            image_bytes (bytes): Image data in bytes format.
            **kwargs: Additional arguments (optional).
        
        Returns:
            bytes: Adjusted image data in bytes format, or None if outlier.
        """
        try:
            sharpness = dataCleaner_38K.calculate_sharpness(image_bytes)
            if sharpness is None:
                return None

            # Remove images with severe blur
            if sharpness < 50:
                return None
            
            img = Image.open(BytesIO(image_bytes))
            if 50 <= sharpness < 70:
                sharpness_factor = 2.0  # Enhance sharpness for moderate blur
            else:
                sharpness_factor = 1.0  # Keep acceptable sharpness as is

            enhancer = ImageEnhance.Sharpness(img)
            adjusted_img = enhancer.enhance(sharpness_factor)
            
            byte_arr = BytesIO()
            adjusted_img.save(byte_arr, format="JPEG")
            return byte_arr.getvalue()
        except Exception as e:
            print(f"Error adjusting sharpness: {e}")
            return None
    @staticmethod
    def align_face(image_bytes, **kwargs):
        """
        Check if image contains a face, return None if no face detected.
        Returns original image bytes if face found (no alignment).
        """
        try:
            img = Image.open(BytesIO(image_bytes))
            
            # Detect faces (MTCNN requires numpy array)
            faces = dataCleaner_38K.detector.detect_faces(np.array(img))
            
            # Return None if no faces detected
            return image_bytes if faces else None
            
        except Exception as e:
            print(f"Face detection failed: {e}")
            return None

    @staticmethod
    def check_resolution(image_bytes, min_resolution=(64, 64)):
        """
        Check if an image meets the minimum resolution requirement.
        
        Parameters:
            image_bytes (bytes): Image data in bytes format
            min_resolution (tuple): Minimum required resolution (width, height) - defaults to (64, 64)
        
        Returns:
            bool: True if meets resolution requirement, False otherwise
        """
        try:
            img = Image.open(BytesIO(image_bytes))
            return (img.size[0] >= min_resolution[0] and 
                    img.size[1] >= min_resolution[1])
        except Exception as e:
            print(f"Error checking resolution: {e}")
            return False
    @staticmethod
    def apply_transformation_to_df(df, transformation_function, **kwargs):
        """
        Apply a transformation function to both `image1` and `image2` columns in the DataFrame.
        Removes rows where either image fails transformation (returns None).
        
        Parameters:
            df (pd.DataFrame): DataFrame containing image data.
            transformation_function (function): Function to apply to the images.
            **kwargs: Additional arguments to pass to the transformation function.
        
        Returns:
            pd.DataFrame: Updated DataFrame with valid transformed images.
        """
        def process_image(x):
            if not x["bytes"]:
                return None  # Mark for removal
                
            try:
                # Extract bytes from BytesIO or use raw bytes
                if isinstance(x["bytes"], BytesIO):
                    byte_data = x["bytes"].getvalue()
                else:
                    byte_data = x["bytes"]
                
                # Apply transformation
                transformed = transformation_function(byte_data, **kwargs)
                
                # Return transformed bytes or None
                return {"bytes": transformed} if transformed else None
                
            except Exception as e:
                print(f"Error processing image: {e}")
                return None  # Mark for removal
    
        # Track original size
        original_size = len(df)
        
        # Apply transformation and filter
        for column in ["image1", "image2"]:
            df[column] = df[column].apply(process_image)
        
        # Remove rows where either image1 or image2 is None
        df = df.dropna(subset=["image1", "image2"])
        
        # Print removal stats
        removed_count = original_size - len(df)
        print(f"Removed {removed_count} instances ({(removed_count / original_size) * 100:.1f}%)")
        
        return df
            
    @staticmethod
    
    def analyze_brightness_distribution(df):
        """
        Computes brightness for all images in the DataFrame and plots the distribution.
        
        Parameters:
            df (pd.DataFrame): DataFrame with 'image1' and 'image2' columns containing image bytes
        
        Returns:
            sorted_brightness (list): List of tuples (image_id, brightness) sorted by brightness
        """
        brightness_scores = {}  # Store brightness values as {image_id: brightness}
        
        # Process both image columns
        for col in ['image1', 'image2']:
            for idx, row in df.iterrows():
                image_bytes = row[col]['bytes']
                image_id = f"{col}_{idx}"  # Unique identifier for tracking
                
                brightness = dataCleaner_38K.calculate_brightness(image_bytes)
                
                if brightness is not None:
                    brightness_scores[image_id] = brightness
    
        # Sort images by brightness
        sorted_brightness = sorted(brightness_scores.items(), key=lambda x: x[1])
        
        # Plot brightness distribution
        plt.figure(figsize=(8, 5))
        plt.hist(list(brightness_scores.values()), bins=30, color='blue', alpha=0.7)
        plt.axvline(x=np.mean(list(brightness_scores.values())), 
                    color='red', 
                    linestyle='dashed', 
                    label=f'Mean: {np.mean(list(brightness_scores.values())):.1f}')
        plt.xlabel("Brightness Level (0-255)")
        plt.ylabel("Number of Images")
        plt.title("Brightness Distribution of Dataset Images")
        plt.legend()
        plt.show()
        
        return sorted_brightness
    
    @staticmethod
    def analyze_contrast_distribution(df):
        """
        Computes contrast for all images and plots the distribution.
        
        Parameters:
            df (pd.DataFrame): DataFrame with image bytes
        
        Returns:
            sorted_contrast (list): Sorted list of (image_id, contrast) tuples
        """
        contrast_scores = {}
        
        for col in ['image1', 'image2']:
            for idx, row in df.iterrows():
                image_bytes = row[col]['bytes']
                image_id = f"{col}_{idx}"
                
                contrast = dataCleaner_38K.calculate_contrast(image_bytes)
                
                if contrast is not None:
                    contrast_scores[image_id] = contrast
    
        # Sort and plot
        sorted_contrast = sorted(contrast_scores.items(), key=lambda x: x[1])
        
        plt.figure(figsize=(8, 5))
        plt.hist(list(contrast_scores.values()), bins=30, color='green', alpha=0.7)
        plt.axvline(x=np.mean(list(contrast_scores.values())), 
                    color='red', 
                    linestyle='dashed',
                    label=f'Mean: {np.mean(list(contrast_scores.values())):.1f}')
        plt.xlabel("Contrast Level")
        plt.ylabel("Number of Images")
        plt.title("Contrast Distribution Analysis")
        plt.legend()
        plt.show()
        
        return sorted_contrast
    
    @staticmethod
    def analyze_sharpness_distribution(df):
        """
        Computes sharpness for all images and plots the distribution.
        
        Parameters:
            df (pd.DataFrame): DataFrame with image bytes
        
        Returns:
            sorted_sharpness (list): Sorted list of (image_id, sharpness) tuples
        """
        sharpness_scores = {}
        
        for col in ['image1', 'image2']:
            for idx, row in df.iterrows():
                image_bytes = row[col]['bytes']
                image_id = f"{col}_{idx}"
                
                sharpness = dataCleaner_38K.calculate_sharpness(image_bytes)
                
                if sharpness is not None:
                    sharpness_scores[image_id] = sharpness
    
        # Sort and plot
        sorted_sharpness = sorted(sharpness_scores.items(), key=lambda x: x[1])
        
        plt.figure(figsize=(8, 5))
        plt.hist(list(sharpness_scores.values()), bins=30, color='purple', alpha=0.7)
        plt.axvline(x=np.mean(list(sharpness_scores.values())), 
                    color='red', 
                    linestyle='dashed',
                    label=f'Mean: {np.mean(list(sharpness_scores.values())):.1f}')
        plt.xlabel("Sharpness Level")
        plt.ylabel("Number of Images")
        plt.title("Sharpness Distribution Analysis")
        plt.legend()
        plt.show()
        
        return sorted_sharpness

    @staticmethod
    def save_df_to_pc(df, filename="cleaned_data38K", format="parquet", path="/content/"):
        """
        Save DataFrame to local PC from Google Colab.
        
        Parameters:
            df (pd.DataFrame): DataFrame to save
            filename (str): Base name for file (without extension)
            format (str): File format - 'parquet' (default) or 'csv'
            path (str): Temporary path in Colab environment
        
        Returns:
            None (triggers file download)
        """
        try:

            full_path = os.path.join(path, f"{filename}.{format}")
            
            # Save in specified format
            if format.lower() == "parquet":
                df.to_parquet(full_path, index=False)
            elif format.lower() == "csv":
                df.to_csv(full_path, index=False)
            else:
                raise ValueError("Unsupported format. Use 'parquet' or 'csv'")
                
            # Trigger download
            files.download(full_path)
            print(f"File '{filename}.{format}' downloaded successfully!")
            
        except Exception as e:
            print(f"Error saving file: {e}")