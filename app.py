from ultralytics import YOLO
import pandas as pd
from PIL import Image
import io
from typing import List

# Load a model
model = YOLO("model/best.pt")  # pretrained YOLO11n model

def get_images_from_bytes(binary_images: List[bytes]) -> List[Image]:
    """Convert image from bytes to PIL RGB format
    
    Args:
        binary_image (bytes): The binary representation of the image
    
    Returns:
        PIL.Image: The image in PIL RGB format
    """
    return [Image.open(binary_image) for binary_image in binary_images]

def post_process(data: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the DataFrame to retain only the highest confidence entry per category.

    Args:
        data (pd.DataFrame): The DataFrame containing detection results.
    
    Returns:
        pd.DataFrame: Processed DataFrame with only the highest confidence detection per category.
    """
    processed_data = data.loc[data.groupby(data['name'].str.split('-').str[0])['confidence'].idxmax()]
    return processed_data[['name', 'confidence']]

def run_inference(images: List[Image]) -> List[pd.DataFrame]:
    """
    Runs YOLO inference on a list of images and returns post-processed results.

    Args:
        images (List[Image]): List of images.

    Returns:
        List[pd.DataFrame]: List of post-processed DataFrames.
    """
    model = YOLO("model/best.pt")  # Load pretrained model
    results = model.predict(images, conf=0.20)

    processed_results = [post_process(result.to_df()).to_dict(orient="records") for result in results]
    print(processed_results)
    return processed_results

def to_json(results: List[pd.DataFrame]) -> List[str]:
    """
    Converts a list of DataFrames to JSON format.

    Args:
        results (List[pd.DataFrame]): List of pandas DataFrames.

    Returns:
        List[str]: List of JSON-serialized dictionaries.
    """
    return [df.to_json(orient='records') for df in results]

# Run batched inference on a list of images
# image_list: List[str] = ["./images/img-3.jpg", "./images/img-2.jpg"]
# results: List[pd.DataFrame] = run_inference(image_list)
# json_results: List[str] = to_json(results)
# print(json_results)
