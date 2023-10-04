import time
import asyncio
import functools

import torch
import numpy as np

from typing import List
from concurrent.futures import ThreadPoolExecutor

from pokesim.utils import preprocess
from pokesim.structs import EnvStep, ActorStep, ModelOutput

_POOL = ThreadPoolExecutor()


class Inference:
    def __init__(
        self,
        model,
        device,
        batch_size: int = None,
        timeout: float = 0.1,
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size or 1024
        self.timeout = timeout

        self.full_queue = asyncio.Queue()

        self.optimized = False if batch_size is None else True
        self.run_in_executor = False

        self.loop = asyncio.get_event_loop()

    async def compute(self, step: EnvStep) -> asyncio.Future:
        fut = self.loop.create_future()
        await self.full_queue.put((step, fut))
        return fut

    async def _get(self):
        if self.optimized:
            item = await self.full_queue.get()
        else:
            item = await asyncio.wait_for(self.full_queue.get(), timeout=self.timeout)
        return item

    async def _run_callback(self):
        last_optim_time = None
        futs = []
        buffer = []
        while True:
            try:
                step, fut = await self._get()
            except:
                if buffer:
                    last_optim_time = time.time()
                    self.batch_size = len(buffer)
                    print(f"Updating `batch_size={self.batch_size}`")

                    yield futs, buffer

                    futs.clear()
                    buffer.clear()
            else:
                futs.append(fut)
                buffer.append(step)

                if len(buffer) >= self.batch_size:
                    if (
                        not self.optimized
                        and last_optim_time
                        and (time.time() - last_optim_time) >= 60
                    ):
                        self.optimized = True
                        print("Using Optimized `await queue.get()`")

                    yield futs, buffer

                    futs.clear()
                    buffer.clear()

    async def run(self):
        outputs = None
        loader = self._run_callback()
        while True:
            batch = await anext(loader)
            futs, buffer = batch
            outputs = await self._process_batch(buffer)
            self._set_outputs(outputs, futs)

    async def _process_batch(self, buffer):
        return await self.loop.run_in_executor(
            _POOL, functools.partial(self._forward_model, buffer)
        )

    def _set_outputs(self, outputs, futs):
        for fut, *output in zip(futs, *(o.squeeze(0) for o in outputs)):
            model_output = ModelOutput(*(arr.cpu().squeeze().numpy() for arr in output))
            fut.set_result(model_output)

    @torch.no_grad()
    def _forward_model(self, buffer: List[EnvStep]) -> ActorStep:
        batch = preprocess(np.stack([step.raw_obs for step in buffer], axis=1))
        batch = {k: torch.from_numpy(v).to(self.device) for k, v in batch.items()}
        mask = torch.from_numpy(
            np.stack([step.legal for step in buffer], axis=1).astype(np.bool_)
        ).to(self.device)
        out = self.model(**batch, mask=mask)
        return out
