import pandas as pd
from PIL import Image
from io import BytesIO
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io
import random
class dataCleaner_38K():
    
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
            
    def remove_corrupted_images_bytes(self,df):
        """
        Remove rows where either `image1` or `image2` is corrupted and return a cleaned Pandas DataFrame.
    
        Parameters:
            df (pd.DataFrame): A DataFrame where each row contains `image1` and `image2` 
            as dictionaries with a "bytes" key.
    
        Returns:
            pd.DataFrame: A cleaned DataFrame with corrupted rows removed.
        """
        # Create a mask to identify valid rows
        mask = df.apply(lambda row: self.check_image_bytes(row["image1"]["bytes"]) and self.check_image_bytes(row["image2"]["bytes"]), axis=1)
    
        # Filter the DataFrame using the mask
        cleaned_df = df[mask]
    
        # Print the number of removed rows
        num_removed = len(df) - len(cleaned_df)
        print(f"Removed {num_removed} corrupted rows.")
    
        return cleaned_df
        
    def visualize_sample(self,df):
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
        
    def process_gray_rgb(self, df):
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
            if self.check_image_mode(row["image1"]["bytes"]) == "L":  # Grayscale
                df.at[idx, "image1"]["bytes"] = self.convert_to_rgb(row["image1"]["bytes"])
                grayCount += 1
            # Process image2
            if self.check_image_mode(row["image2"]["bytes"]) == "L":  # Grayscale
                df.at[idx, "image2"]["bytes"] = self.convert_to_rgb(row["image2"]["bytes"])
                grayCount += 1
        print(f"{grayCount} img out of {38000*2} were gray scale.")
        return df