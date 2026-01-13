#https://huggingface.co/docs/transformers/en/tasks/zero_shot_object_detection
import yaml
from pathlib import Path
from datetime import datetime
from src.utils import create_video_from_images, object_detection

def get_paths_from_config(config: dict) -> tuple[Path, Path, Path]:
    """Extract paths from configuration dictionary."""

    raw_input = config.get("input_path")
    if raw_input is None:
        raise ValueError("config missing 'input_path'")

    raw_path = Path(raw_input)

    if raw_path.exists() and raw_path.is_dir():
        files = sorted([p for p in raw_path.iterdir() if p.is_file()])
        # store all file paths (as strings) in config for later use
        config["input_path_list"] = [str(p) for p in files]
    else:
        # single path (file or non-existent) stored as single-item list
        config["input_path_list"] = [str(raw_path)]
    input_path = Path(config['input_path'])
    output_dir = Path(config['output_dir'])
    output_name = config.get('output_name')
    task = config.get('task')
    frames_dir = Path(config.get('frames_dir'))

    output_dir = output_dir.joinpath(task)
    output_dir.mkdir(parents=True, exist_ok=True)

    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    # create dated frames root and a subfolder per input file (by base name)
    # target_root = frames_dir.joinpath(date)
    target_root = frames_dir
    target_root.mkdir(parents=True, exist_ok=True)

    input_list = config.get("input_path_list", [])
    for p in input_list:
        subfolder = target_root.joinpath(Path(p).stem)
        subfolder.mkdir(parents=True, exist_ok=True)

    # store mapping for later use if needed
    config["frames_subdirs"] = [str(target_root.joinpath(Path(p).stem)) for p in input_list]
    # frames_dir = frames_dir.joinpath(date)
    # frames_dir.mkdir(parents=True, exist_ok=True)

    if output_name:
        output_path = output_dir.joinpath(output_name)
    else:
        output_path = output_dir.joinpath(input_path.name)

    return input_list, output_path, config["frames_subdirs"]


def main():

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    input_path_list, output_path, frames_subdirs = get_paths_from_config(config)

    for input_path, frames_dir in zip(input_path_list, frames_subdirs):
        object_detection(str(input_path), str(frames_dir), config)

    # path_video_frame_dirs = [config['frames_dir']+'/'+dir for dir in os.listdir(config['frames_dir'])]
    # output_path =  [config['output_dir']+'/'+config['task']+'/'+dir+'.mp4' for dir in os.listdir(config['frames_dir'])]

    # for frames_dir, output_file in zip(path_video_frame_dirs, output_path):
    #     print(f"Creating video from {frames_dir} -> {output_file}")
    #     create_video_from_images(str(frames_dir), str(output_file), fps=30)

if __name__ == "__main__":
    main()
