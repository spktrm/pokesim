import os
import torch
from pokesim.nn.modelv2 import Model
from pokesim.rnad.actor import run_environment


class FillerQueue:
    def put(self, *args, **kwargs):
        pass


def get_most_recent_file(dir_path):
    # List all files in the directory
    files = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f))
    ]

    if not files:
        return None

    # Sort files by creation time
    most_recent_file = max(files, key=os.path.getctime)

    return most_recent_file


def main(worker_index):
    fpath = get_most_recent_file("ckpts")
    print(fpath)

    ckpt = torch.load(fpath, map_location="cpu")
    state_dict = ckpt["params"]

    model = Model()
    model.load_state_dict(state_dict)

    filler_queue = FillerQueue()
    run_environment(
        worker_index, model, None, filler_queue, filler_queue, verbose=True, threshold=1
    )


if __name__ == "__main__":
    main(1000)
