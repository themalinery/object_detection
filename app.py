import gradio as gr
import tempfile
import shutil
from pathlib import Path
import sys

# Monkey patch to fix gradio_client schema parsing bug
try:
    from gradio_client import utils as client_utils
    
    # Patch the main function that fails when encountering booleans
    original_json_schema_to_python_type = client_utils._json_schema_to_python_type
    
    def patched_json_schema_to_python_type(schema, defs=None):
        # If schema is a boolean, return a simple dict representation
        if isinstance(schema, bool):
            return "bool"
        return original_json_schema_to_python_type(schema, defs)
    
    client_utils._json_schema_to_python_type = patched_json_schema_to_python_type
except Exception as e:
    print(f"Warning: Could not apply monkey patch: {e}")

from src.utils import create_video_from_images, object_detection

def process_video(video_file, labels_text, frame_color):
    # Parse labels
    text_labels = [label.strip() for label in labels_text.split(',') if label.strip()]
    
    if not text_labels:
        return None
    
    # Create config
    config = {
        'labels': text_labels,
        'frame_colour': frame_color
    }
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        frames_dir = temp_path / "frames"
        frames_dir.mkdir()
        output_video = temp_path / "output.mp4"
        
        # Process video to frames
        object_detection(str(video_file), str(frames_dir), config)
        
        # Create video from frames
        create_video_from_images(str(frames_dir), str(output_video), fps=30)
        
        # Copy to a permanent location for download
        results_dir = Path("./results/gradio_outputs")
        results_dir.mkdir(parents=True, exist_ok=True)
        final_output = results_dir / f"detected_{Path(video_file).stem}.mp4"
        shutil.copy(output_video, final_output)
        
        return str(final_output)

# Simple gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Video Object Detection")
    
    with gr.Row():
        video_input = gr.File(label="Upload Video", file_types=[".mp4", ".avi", ".mov"])
        labels_input = gr.Textbox(label="Labels (comma-separated)", placeholder="cat,dog,person")
        color_input = gr.ColorPicker(label="Bounding Box Color", value="#FF0000")
    
    process_btn = gr.Button("Process")
    output = gr.File(label="Download")
    
    def process(video, labels, color):
        if not video or not labels:
            return None
        try:
            result = process_video(video, labels, color)
            return result
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    process_btn.click(process, inputs=[video_input, labels_input, color_input], outputs=output)

if __name__ == "__main__":
    demo.launch(share=True)
