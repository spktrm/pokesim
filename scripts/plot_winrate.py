import wandb
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
            random_values.rolling(window).mean(),
            default_values.rolling(window).mean(),
        ],
        axis=1,
    )[window:]
    fig = ((concat_df + 1) / 2).plot(height=800)
    fig.show()


if __name__ == "__main__":
    run_id = "jtwin/pokesim/w8erff3y"
    window = 1000

    main(run_id, window)
