import os
from moviepy import ImageSequenceClip
from natsort import natsorted
from transformers import pipeline
from transformers.image_utils import load_image
from PIL import ImageDraw, Image, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
import cv2
import os

def create_video_from_images(folder_path, output_video_file, fps):
    """
    Creates a video file from a sequence of images in a folder.

    Args:
        folder_path (str): The path to the folder containing the images.
        output_video_file (str): The name of the output video file (e.g., 'my_video.mp4').
        fps (int): The frames per second for the output video.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    # List all image files in the folder.
    # We use natsorted to ensure files with numerical names (e.g., image-1.png, image-10.png)
    # are sorted in a human-friendly way.
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = [
        os.path.join(folder_path, f)
        for f in natsorted(os.listdir(folder_path))
        if f.lower().endswith(supported_extensions)
    ]

    if not image_files:
        print(f"Error: No supported image files found in '{folder_path}'.")
        return

    if len(image_files) < 2:
        print("Error: At least two images are required to create a video.")
        return

    print(f"Found {len(image_files)} images. Creating video...")

    try:
        # Create a video clip from the list of image files.
        clip = ImageSequenceClip(image_files, fps=fps)

        # Write the video file to the specified path.
        clip.write_videofile(output_video_file, fps=fps)

        print(f"Successfully created video: '{output_video_file}'")
    except Exception as e:
        print(f"An error occurred while creating the video: {e}")


def object_detection(path_video, output_folder, config):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_labels = config.get('labels', [])
    frame_color = config.get('frame_colour')

    checkpoint = "iSEE-Laboratory/llmdet_tiny"  #"openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det"
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
            draw.rectangle((xmin, ymin, xmax, ymax), outline=frame_color, width=10)
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