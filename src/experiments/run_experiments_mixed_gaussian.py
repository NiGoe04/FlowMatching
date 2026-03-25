from src.experiments.mixed_gaussian_framework.run_experiments_mixed_gaussian import run


if __name__ == "__main__":
    report = run()
    print(f"Saved report to: {report}")
