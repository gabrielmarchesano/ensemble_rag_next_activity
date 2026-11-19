from pathlib import Path
import random

import old_preprocessing as op


ROOT_DIR = Path(__file__).resolve().parent
LOGS_DIR = ROOT_DIR / "logs"
OUTPUT_DIR = ROOT_DIR / "tests" / "original_logparser_output"
TEST_PROPORTION = 0.3
RANDOM_SEED = 42


def main():
    log_files = sorted(LOGS_DIR.glob("*.xes"))
    if not log_files:
        raise FileNotFoundError(f"No .xes event logs found in {LOGS_DIR}")

    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for log_file in log_files:
        print(f"\nParsing {log_file.name}...")
        content = op.read_event_log(log_file.name)
        traces = op.extract_traces(content)
        del content

        train_traces, test_traces = op.split_traces(traces, TEST_PROPORTION)
        train_path = OUTPUT_DIR / f"train_set_{log_file.stem}.csv"
        test_path = OUTPUT_DIR / f"test_set_{log_file.stem}.csv"

        op.generate_csv_from_traces(
            train_traces,
            str(train_path),
            log_name=log_file.name,
        )
        op.generate_csv_from_test_set(
            test_traces,
            str(test_path),
            size=None,
            log_name=log_file.name,
        )

        print(
            f"{log_file.name}: {len(traces)} traces parsed "
            f"({len(train_traces)} train, {len(test_traces)} test); "
            f"CSV files written to {OUTPUT_DIR}"
        )


if __name__ == "__main__":
    main()