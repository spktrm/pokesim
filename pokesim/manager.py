import asyncio
import multiprocessing as mp

from pokesim.structs import EnvStep, TimeStep, Trajectory
from pokesim.actor import Actor, EvalActor, SelfplayActor
from pokesim.constants import _NUM_HISTORY


class Manager:
    def __init__(
        self,
        worker_index: int,
        process: asyncio.subprocess.Process,
        actor1: Actor,
        actor2: Actor,
    ):
        self.worker_index = worker_index
        self.process = process
        self.actor1 = actor1
        self.actor2 = actor2
        self.is_eval = isinstance(actor1, EvalActor) or isinstance(actor2, EvalActor)

    async def run(self, progress_queue: mp.Queue, learn_queue: mp.Queue = None):
        traj = {}
        dones = {}
        hist = {0: [], 1: []}

        actor: Actor
        while True:
            data = await self.process.stdout.readuntil(b"\n\n")

            env_step = EnvStep.from_data(data)

            player_id = env_step.player_id.item()

            try:
                actor: Actor = getattr(self, f"actor{player_id + 1}")
            except Exception as e:
                print(e)
                continue

            hist[player_id].append(env_step)
            env_step = EnvStep.from_stack(hist[player_id][-_NUM_HISTORY:])

            done = 1 - env_step.valid
            game_id = env_step.game_id.item()

            if game_id not in dones:
                dones[game_id] = 0

            dones[game_id] += done
            if game_id not in traj:
                traj[game_id] = []

            if not done:
                if isinstance(actor, SelfplayActor):
                    _, actor_step1 = await actor.choose_action(
                        env_step, policy_select=0
                    )
                    policy_select = 1 if actor_step1.action.item() == 0 else 2
                    action, actor_step2 = await actor.choose_action(
                        env_step, policy_select=policy_select
                    )
                    if learn_queue is not None:
                        for actor_step in [actor_step1, actor_step2]:
                            time_step = TimeStep(
                                id=game_id,
                                actor=actor_step,
                                env=env_step,
                            )
                            traj[game_id].append(time_step)
                else:
                    action, actor_step = await actor.choose_action(env_step)

                self.process.stdin.write(action)
                await self.process.stdin.drain()

            if dones[game_id] >= 2:
                if learn_queue is not None:
                    time_step = TimeStep(
                        id=game_id,
                        actor=actor_step,
                        env=env_step,
                    )
                    traj[game_id].append(time_step)
                    trajectory = Trajectory.from_env_steps(traj[game_id])
                    learn_queue.put(trajectory.serialize())
                    progress_queue.put(len(traj[game_id]))

                assert abs(env_step.rewards).sum() > 0
                for player_id, actor in enumerate([self.actor1, self.actor2]):
                    reward = env_step.rewards[..., player_id]
                    actor._done_callback(reward.item())

                dones[game_id] = 0
                traj[game_id] = []
                hist = {0: [], 1: []}
