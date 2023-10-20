import wandb
import argparse
import pandas as pd


def main(run_id: str, window: int, num_samples: int = int(200e3)):
    pd.options.plotting.backend = "plotly"
    api = wandb.Api(timeout=60)
    run = api.run(run_id)
    df = run.history(samples=num_samples)

    rows = []

    for column in {"prev_r", "random_r", "default_r"}:
        try:
            values = df[column].dropna().reset_index().drop(["index"], axis=1)
            rows.append(values)
        except:
            pass

    concat_df = pd.concat(
        [((values + 1) / 2).rolling(window).mean() for values in rows],
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
