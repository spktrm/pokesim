import torch

from pokesim.nn.modelv2 import Model
from pokesim.rnad.actor import run_environment
from pokesim.utils import get_most_recent_file


class FillerQueue:
    def put(self, *args, **kwargs):
        pass


def main(worker_index):
    # fpath = get_most_recent_file("ckpts")
    # print(fpath)

    # ckpt = torch.load(fpath, map_location="cpu")
    # state_dict = ckpt["params"]

    model = Model()
    # model.load_state_dict(state_dict, strict=False)

    filler_queue = FillerQueue()
    run_environment(
        worker_index,
        model,
        None,
        filler_queue,
        filler_queue,
        None,
        verbose=True,
        threshold=1,
    )


if __name__ == "__main__":
    main(100)
