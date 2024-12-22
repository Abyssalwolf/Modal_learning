import modal

app=modal.App("custom-container")

image = modal.Image.debian_slim()
image = image.apt_install("ffmpeg")
image = image.pip_install("transformers[torch]")
image = image.pip_install("ffmpeg")

@app.function(gpu="A10G", image=image)
def check_cuda():
    import torch
    import transformers

    print("Transformers version: ",transformers.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())

@app.function(gpu="A10G", image=image)
def transcribe_audio(file_url: str):
    from transformers import pipeline
    transcriber = pipeline(model="openai/whisper-tiny.en", device="cuda")
    result = transcriber(file_url)
    print(result["text"])

@app.local_entrypoint()
def main():
    check_cuda.remote()
    transcribe_audio.remote("https://modal-cdn.com/mlk.flac")