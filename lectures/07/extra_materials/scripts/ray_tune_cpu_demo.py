import os

import ray
from ray import tune


def trainable(config):
    x = config["x"]
    y = config["y"]
    score = (x - 3) ** 2 + (y + 1) ** 2
    tune.report(loss=score)


def main() -> None:
    ray_address = os.getenv("RAY_ADDRESS", "ray://ray-head:10001")
    # For local Docker network in this setup, classic address is enough.
    fallback = "ray-head:6379"

    try:
        ray.init(address=ray_address)
    except Exception:
        ray.init(address=fallback)

    tuner = tune.Tuner(
        trainable,
        param_space={
            "x": tune.uniform(-10, 10),
            "y": tune.uniform(-10, 10),
        },
        tune_config=tune.TuneConfig(num_samples=8, metric="loss", mode="min"),
    )

    results = tuner.fit()
    best = results.get_best_result(metric="loss", mode="min")
    print("Best config:", best.config)
    print("Best loss:", best.metrics["loss"])

    ray.shutdown()


if __name__ == "__main__":
    main()
