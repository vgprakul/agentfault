from injector.generator import DatasetGenerator


def main():
    generator = DatasetGenerator(
        baseline_dir="data/trajectories/baseline",
        output_dir="data/trajectories/injected",
        seed=42,
    )

    generator.generate(
        target_count=500,
    )


if __name__ == "__main__":
    main()
