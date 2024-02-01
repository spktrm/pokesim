from modal import Image, Mount, Secret, Stub


stub = Stub("pokesim")
image = (
    Image.debian_slim()
    .apt_install(["nodejs", "npm", "wget", "curl"])
    .run_commands(["npm install -g n ", "n latest", "npm install typescript -g"])
    .run_commands("git clone https://github.com/spktrm/pokesim /pokesim")
    .workdir("/pokesim")
    .run_commands(
        [
            "git checkout feature/add_history",
            "npm install",
            "tsc",
            "pip install -r requirements.txt",
        ]
    )
    .pip_install(["wandb", "tabulate"])
    .run_commands(["export PYTHONPATH=/pokesim"])
)


@stub.function(
    # cpu=20,
    # gpu="any",
    image=image,
    secret=Secret.from_name("my-wandb-secret"),
)
def train():
    import os
    import sys
    import subprocess

    print(os.listdir())

    sys.path.append("/pokesim")

    # # args = ["make"]
    # # proc = subprocess.Popen(args)

    args = ["python", "pokesim/rnad/main.py"]
    proc = subprocess.Popen(args)

    proc.wait()


@stub.local_entrypoint()
def start():
    train.remote()
