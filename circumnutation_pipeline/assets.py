import os
from pathlib import Path
import yaml
import json
import hashlib
import h5py
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sleap
import shutil
import imageio as iio
from sleap import Labels
import logging
from circumnutation_pipeline.utils import (
    generate_run_hash,
    process_image_directory,
    create_videos,
)


import dagster as dg
from dagster import (
    Definitions,
    graph,
    job,
    resource,
    op,
    asset,
    AssetOut,
    AssetIn,
    In,
    Out,
    Output,
)


@resource(config_schema={"pipeline_config_path": str})
def pipeline_config_resource(context):
    config_path = Path(context.resource_config["pipeline_config_path"])

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


@op(required_resource_keys={"pipeline_config"})
def initialize_run(context):

    pipeline_config = context.resources.pipeline_config

    experiment_name = pipeline_config["setup"]["experiment_name"]
    plate_number = pipeline_config["setup"]["plate_number"]
    treatment = pipeline_config["setup"]["treatment"]

    raw_images_path = Path(pipeline_config["path"]["raw_images"])
    labels_path = Path(pipeline_config["path"]["labels"])

    run_id = generate_run_hash()
    logging.info(f"Generated run hash: {run_id}")

    run_path = Path.cwd() / "runs" / run_id
    logging.info(f"Run path: {run_path}")

    paths = [
        run_path,
        raw_images_path,
        run_path / "configs",
        run_path / "h5_videos",
        run_path / "videos",
        run_path / "labels",
        run_path / "models",
        run_path / "predictions",
        run_path / "analysis",
    ]

    for path in paths:
        os.makedirs(path, exist_ok=True)
        logging.info(f"Created directory {path}")

    configs_path = Path.cwd() / "circumnutation_pipeline" / "configs"
    configs_list = os.listdir(configs_path)

    for config in configs_list:
        shutil.copy(configs_path / config, run_path / "configs")
        logging.info(f"Saved copy of {config} to {run_path / 'configs'}")

    context.log.info(f"Run {run_id} initialized successfully.")

    return {
        "run_id": run_id,
        "run_path": str(run_path),
    }


@op(
    required_resource_keys={"pipeline_config"},
    ins={"run_info": In()},
    out={"h5_greyscale": Out(str), "h5_color": Out(str)},
)
def create_h5(context, run_info):

    pipeline_config = context.resources.pipeline_config

    run_id = run_info["run_id"]
    run_path = Path(run_info["run_path"])
    raw_images_path = Path(pipeline_config["path"]["raw_images"])
    h5_videos_path = run_path / "h5_videos"

    logging.info(f"Creating greyscale h5 video from images directory {raw_images_path}")
    process_image_directory(
        source_dir=raw_images_path,
        output_dir=h5_videos_path,
        experiment_name=pipeline_config["setup"]["experiment_name"],
        treatment=pipeline_config["setup"]["treatment"],
        num_plants=pipeline_config["setup"]["num_plants"],
        greyscale=True,
    )
    logging.info(
        f"Finished creating greyscale h5 video from images directory {raw_images_path}"
    )

    logging.info(f"Creating color h5 video from images directory {raw_images_path}")
    process_image_directory(
        source_dir=raw_images_path,
        output_dir=h5_videos_path,
        experiment_name=pipeline_config["setup"]["experiment_name"],
        treatment=pipeline_config["setup"]["treatment"],
        num_plants=pipeline_config["setup"]["num_plants"],
        greyscale=False,
    )
    logging.info(
        f"Finished creating color h5 video from images directory {raw_images_path}"
    )

    logging.info(f"Finished creating h5 videos")

    greyscale_h5_path = (
        h5_videos_path
        / f"plate_{pipeline_config['setup']['plate_number']}_greyscale.h5"
    )
    color_h5_path = (
        h5_videos_path / f"plate_{pipeline_config['setup']['plate_number']}_color.h5"
    )

    yield Output(str(greyscale_h5_path), output_name="h5_greyscale")
    yield Output(str(color_h5_path), output_name="h5_color")


@op(
    required_resource_keys={"pipeline_config"},
    ins={"run_info": In(), "h5_color": In()},
    out={"video_path": Out(str)},
)
def create_timelapse_videos(context, run_info, h5_color):

    logging.info(f"Processing color h5 video in location {h5_color}")

    pipeline_config = context.resources.pipeline_config

    create_videos(
        h5_color, pipeline_config["setup"]["plate_number"], run_info["run_id"]
    )

    create_videos(
        h5_color,
        pipeline_config["setup"]["plate_number"],
        run_info["run_id"],
        labels=pipeline_config["path"]["labels"],
    )

    return "hello"


####################


# Graph
@graph
def circumnutation_pipeline():

    # TODO: If pipeline_config plate number is not the same as plate number in raw
    # images filename, raise error

    # TODO: If run folder exists but overwrite duplicate exists, terminate pipeline

    # TODO: If metadata csv in process_image_directory
    # exists in output directory, don't overwrite

    # TODO: Save copies of all configs to the run_id directory

    # TODO: Update output to create_timelapse_videos

    # TODO: Check for operating system and run appropriate .sh or .bat script
    # use platform module -> platform.system() = Darwin or Windows + integrate with wandb

    run_info = initialize_run()
    h5_greyscale, h5_color = create_h5(run_info)
    create_timelapse_videos(run_info=run_info, h5_color=h5_color)


# Job
circumnutation_job = circumnutation_pipeline.to_job(
    resource_defs={"pipeline_config": pipeline_config_resource}
)

# Definitions
defs = Definitions(
    jobs=[circumnutation_job], resources={"pipeline_config": pipeline_config_resource}
)
