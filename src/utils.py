import os
from moviepy import ImageSequenceClip
from natsort import natsorted

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