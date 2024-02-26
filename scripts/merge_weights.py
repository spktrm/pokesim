import torch

from typing import Dict

from pokesim.nn.model import Model
from pokesim.utils import get_most_recent_file


StateDict = Dict[str, torch.Tensor]


def copy_weights(old: StateDict, new: StateDict) -> StateDict:
    for key, old_param in old.items():
        try:
            new_param = new[key]
            new_param[..., : old_param.shape[-1]] = old_param
            new[key] = new_param
        except:
            pass

    return new


def main():
    fpath = get_most_recent_file("ckpts")
    print(fpath)
    init = torch.load(fpath, map_location="cpu")

    for key, value in init.items():
        if key.startswith("params"):
            old_model_params = init[key]
            new_model_params = Model().state_dict()

            new_model_params = copy_weights(old_model_params, new_model_params)

            init[key] = new_model_params

    torch.save(init, fpath.split(".")[0] + "-merged.pt")


if __name__ == "__main__":
    main()
