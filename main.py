#https://huggingface.co/docs/transformers/en/tasks/zero_shot_object_detection
from transformers import pipeline
from transformers.image_utils import load_image
from PIL import ImageDraw, Image, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
import yaml
from pathlib import Path
from datetime import datetime
from src.utils import create_video_from_images
import cv2
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

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


def object_detection(path_video, output_folder):
    checkpoint = "iSEE-Laboratory/iSEE-Laboratory_llmdet_large"  #"openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det"
    model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint, device_map="auto")
    processor = AutoProcessor.from_pretrained(checkpoint)

    # Initialize video capture
    vidcap = cv2.VideoCapture(path_video)

    frame_count = 0
    # Initialize hand tracking
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if not ret:
            break

        print(f"Processing frame {frame_count}")

        # Convert the BGR image to RGB and ensure RGB mode
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame).convert("RGB")

        # use a flat list of labels for single-image inference
        text_labels = ["cat"]
        inputs = processor(text=text_labels, images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # monkeypatch ImageDraw.text to accept a `fontsize` argument (absolute pixels or fraction of image height)


        results = processor.post_process_grounded_object_detection(
                    outputs, threshold=0.50, target_sizes=[(image.height, image.width)])[0]

        draw = ImageDraw.Draw(image)

        scores = results.get("scores", [])
        text_labels_res = results.get("text_labels", [])
        boxes = results.get("boxes", [])

        for box, score, text_label in zip(boxes, scores, text_labels_res):
            xmin, ymin, xmax, ymax = box
            draw.rectangle((xmin, ymin, xmax, ymax), outline="white", width=10)
            # convert score to float safely
            try:
                score_val = float(score)
            except Exception:
                score_val = round(score.item(), 2)

            # font_size = max(10, int(0.1 * image.height))  # 10% of image height, minimum 10 pixels
            #font = ImageFont.load_default(size=80)
            font = ImageFont.truetype("fonts/Perfect DOS VGA 437.ttf", size=60)
            draw.text((xmin, ymin), f"{text_label}: {round(score_val,2)}", fill="black", stroke_width=1, stroke_fill="black", font=font)
        # save the annotated image (PIL image is modified in-place)
        image.save(f"{output_folder}/{frame_count}.png")

        # Exit loop by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        if frame_count == 90:  # limit to first 30 frames
            break

    # Release the video capture and close windows
    vidcap.release()
    cv2.destroyAllWindows()


def main():

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    input_path_list, output_path, frames_subdirs = get_paths_from_config(config)

    for input_path, frames_dir in zip(input_path_list, frames_subdirs):
        object_detection(str(input_path), str(frames_dir))

    # path_video_frame_dirs = [config['frames_dir']+'/'+dir for dir in os.listdir(config['frames_dir'])]
    # output_path =  [config['output_dir']+'/'+config['task']+'/'+dir+'.mp4' for dir in os.listdir(config['frames_dir'])]

    # for frames_dir, output_file in zip(path_video_frame_dirs, output_path):
    #     print(f"Creating video from {frames_dir} -> {output_file}")
    #     create_video_from_images(str(frames_dir), str(output_file), fps=30)

if __name__ == "__main__":
    main()
