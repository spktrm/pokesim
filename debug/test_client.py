import torch
from pokesim.model import Model
from pokesim.rnad.actor import run_environment


class FillerQueue:
    def put(self, *args, **kwargs):
        pass


def main(worker_index):
    ckpt = torch.load(f"ckpts/018562.pt", map_location="cpu")
    state_dict = ckpt["params"]

    model = Model()
    model.load_state_dict(state_dict)

    filler_queue = FillerQueue()
    run_environment(worker_index, model, None, filler_queue, filler_queue, verbose=True)


if __name__ == "__main__":
    main(1000)
