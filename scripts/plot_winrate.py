import wandb
import argparse
import pandas as pd


def main(run_id: str, window: int, num_samples: int = int(200e3)):
    pd.options.plotting.backend = "plotly"
    api = wandb.Api(timeout=60)
    run = api.run(run_id)
    df = run.history(samples=num_samples)

    random_values = df["random_r"].dropna().reset_index().drop(["index"], axis=1)
    default_values = df["default_r"].dropna().reset_index().drop(["index"], axis=1)

    concat_df = pd.concat(
        [
            ((random_values + 1) / 2).rolling(window).mean(),
            ((default_values + 1) / 2).rolling(window).mean(),
        ],
        axis=1,
    )[window:]
    fig = concat_df.plot(height=800)
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, type=str, help="The ID of the run.")
    parser.add_argument("--window", type=int, default=100, help="The window size.")
    args = parser.parse_args()
    main(args.run_id, args.window)
