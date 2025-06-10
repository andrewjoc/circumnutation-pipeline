import pandas as pd
import numpy as np
import h5py
from pathlib import Path
import imageio as iio
import re
import os
import json
import pandas as pd
import hashlib
import yaml
import glob
import logging


def generate_run_hash():
    # Read pipeline config as a dict
    cwd = Path.cwd() / "circumnutation_pipeline"
    with open(cwd / "pipeline_config.yaml", "r") as file:
        try:
            yaml_data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            logging.log(f"Error reading YAML file: {e}")

    # Serialize to a JSON string with sorted keys
    serialized = json.dumps(yaml_data, sort_keys=True)

    # Compute SHA-256 hash
    return hashlib.sha256(serialized.encode()).hexdigest()


def natural_sort(l):
    """Sort a list of strings in a way that considers numerical values within the strings.

    For example, natural_sort(["img2.png", "img10.png", "img1.png"])
    will return ["img1.png", "img2.png", "img10.png"].

    Args:
        l (list): List of strings to sort.

    Returns:
        list: List of sorted strings.
    """
    l = [x.as_posix() if isinstance(x, Path) else x for x in l]
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def process_image_directory(
    source_dir, output_dir, experiment_name, treatment, num_plants, greyscale=True
):
    """
    Processes a directory of images for a plate over time into an h5 file and a metadata file.

    Args:
        source_dir (Path): Path to the source directory containing images.
        output_dir (Path): Path to the directory to store the h5 file and metadata file.
        experiment_name (str):
        treatment (str): The chemical or physical alterations to the plate media.
        num_plants (int): The number of plants expected on a plate image.
        greyscale (bool): Whether or not to convert images to greyscale.
        metadata (bool): Whether or not to save a dataframe of metadata.
    """

    # Check if the source directory exists
    if not source_dir.exists():
        logging.error(f"Source directory {source_dir} does not exist.")
        return

    # Check if the source directory is a directory
    if not source_dir.is_dir():
        logging.error(f"Source path {source_dir} is not a directory.")

    # Make sure output_dir exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files in the source directory
    image_files = list(source_dir.glob("*.tif"))

    if not image_files:
        logging.error(f"No image files found in {source_dir}.")
        return

    logging.info(f"Found {len(image_files)} image files in {source_dir}.")

    # Sort the image files naturally
    image_files = natural_sort(image_files)

    images = []
    df_rows = []

    # Process each image file
    logging.info("Reading images...")
    frame_idx = 0
    for img_file in image_files:

        logging.info(f"Reading {img_file}...")
        # Read the image
        img = iio.imread(img_file)

        if greyscale:
            # Convert image to greyscale
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

            # Convert image to greyscale
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
            img = img.astype(np.uint8)

        # Extract plate number, day, and time from the filename
        # Assuming the filename format is "_set1_day1_timestamp_platenumber.tif"

        timestamp = img_file.split("_")[-2]
        plate_number = img_file.split("_")[-1].split(".")[0]
        filename = img_file.split("/")[-1]

        df_rows.append(
            {
                "experiment": experiment_name,
                "filename": filename,
                "treatment": treatment,
                "plate_number": plate_number,
                "expected_num_plants": num_plants,
                "timestamp": timestamp,
                "frame": frame_idx,
            }
        )

        images.append(img)
        frame_idx += 1

    logging.info("Finished reading images.")

    logging.info("Stacking images...")
    vol = np.stack(images, axis=0)
    logging.info("Finished stacking images.")

    if greyscale:
        h5_name = f"plate_{plate_number}_greyscale.h5"
    else:
        h5_name = f"plate_{plate_number}_color.h5"

    h5_path = output_dir / h5_name

    # Save the volume as a .h5 file
    with h5py.File(h5_path, "w") as f:
        # Create a dataset in the HDF5 file
        logging.info(f"Creating dataset in {h5_name}...")

        if greyscale:
            # Expand dimensions to (frames, height, width, 1)
            vol = np.expand_dims(vol, axis=-1)

            logging.info(f"Vol shape: {vol.shape}")

            f.create_dataset("vol", data=vol, compression="gzip", compression_opts=1)
        else:
            # Expected shape: (frames, height, width, 3)
            logging.info(f"Vol shape: {vol.shape}")

            f.create_dataset("vol", data=vol, compression="gzip", compression_opts=1)

        logging.info(f"Saved vol to {h5_path}")

    # Save the DataFrame to a .csv file
    df_path = output_dir / ("plate_" + plate_number + "_metadata.csv")
    df = pd.DataFrame.from_records(df_rows)
    df.to_csv(df_path, index=False)
    logging.info(f"Saved DataFrame {df_path} to {df_path}")
