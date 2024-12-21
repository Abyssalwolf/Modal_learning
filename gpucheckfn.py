import modal

app = modal.App('gpucheckfn')

@app.function(gpu="A10G")
def check_gpus():
    import subprocess

    print("Here's my GPU:")
    try:
        subprocess.run(["nvidia-smi", "--list-gpus"], check=True)
    except Exception:
        print("No GPU found")

@app.local_entrypoint()
def main():
    print("hello from the .local playground!")
    check_gpus.local()

    print("Let's try this .remote-ly on Modal...")
    check_gpus.remote()