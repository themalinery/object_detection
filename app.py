import gradio as gr
import tempfile
import shutil
from pathlib import Path

from src.utils import create_video_from_images, object_detection

def process_video(video_file, labels_text, frame_color):
    # Parse labels
    text_labels = [label.strip() for label in labels_text.split(',') if label.strip()]
    
    if not text_labels:
        raise gr.Error("Please enter at least one label")
    
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

# Gradio interface
with gr.Blocks(title="Video Object Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Video Object Detection")
    gr.Markdown("Upload a video, enter labels to detect, choose frame color, and download the processed video.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")
            labels_input = gr.Textbox(label="Detection Labels (comma-separated)", placeholder="e.g., cat, dog, person")
            color_input = gr.ColorPicker(label="Bounding Box Color", value="#FF0000")
            process_btn = gr.Button("Process Video", variant="primary")
        
        with gr.Column():
            # Output section
            gr.Markdown("## Output")
            
            output_video = gr.Video(label="Processed Video", interactive=False)
            
            download_button = gr.File(label="Download Processed Video", visible=False)
    
    # Handle processing
    def process_and_update(video, labels_text, frame_color):
        try:
            # Update status
            gr.Info("Processing video... This may take a few minutes.")
            
            output_path = process_video(video, labels_text, frame_color)
            
            gr.Info("Video processing complete!")
            
            return output_path, output_path
        except Exception as e:
            raise gr.Error(f"Processing failed: {str(e)}")
    
    process_btn.click(
        fn=process_and_update,
        inputs=[video_input, labels_input, color_input],
        outputs=[output_video, download_button]
    )

if __name__ == "__main__":
    demo.launch()