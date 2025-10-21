
import argparse
import json
from pathlib import Path

def parse_args():
    """
    Parses command-line arguments.

    Example usage:
    
    # Create a new split from multiple recording directories
    python prepare_tau2_data.py \
      --input-dirs recordings/run_20251014-* \
      --output-dir recordings/processed_for_cookbook \
      --test-fraction 0.2

    # Reuse an existing split for a new set of recordings
    python prepare_tau2_data.py \
      --input-dirs recordings/new_model_run_* \
      --output-dir recordings/processed_for_cookbook_new_model \
      --split-manifest recordings/processed_for_cookbook/split_manifest.json

    # Map input directories to specific run names for grouping
    python prepare_tau2_data.py \
        --input-dirs recordings/llama3-run-* recordings/mistral-run-* \
        --output-dir recordings/processed_for_cookbook \
        --run-name-map llama3-run-1:llama3-70b mistral-run-1:mistral-7b \
        --test-fraction 0.2
    """
    p = argparse.ArgumentParser(
        description="Process tau2 dialogs for cookbook-internal SFT/RFT workflows."
    )
    p.add_argument(
        "--input-dirs",
        type=str,
        required=True,
        nargs="+",
        help="One or more paths to tau2 recording directories (e.g., recordings/run_...).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to save processed files (e.g., /mnt/datasets/tau2-airline).",
    )
    p.add_argument(
        "--run-name-map",
        type=str,
        nargs="*",
        default=[],
        help="Mapping from input directory basename to a desired run name, in 'dirname:run_name' format. "
             "Allows grouping multiple input directories under a single identifier. "
             "If not provided for a directory, its basename is used.",
    )
    
    # Mutually exclusive group for defining the split
    split_group = p.add_mutually_exclusive_group(required=True)
    split_group.add_argument(
        "--test-fraction",
        type=float,
        help="Fraction of tasks to allocate to the test set (e.g., 0.2 for 20%).",
    )
    split_group.add_argument(
        "--split-manifest",
        type=str,
        help="Path to an existing split_manifest.json to reuse a previous split.",
    )
    
    return p.parse_args()

def find_dialog_files(input_dirs: list[str]) -> list[Path]:
    """Finds all tau2_dialogs.jsonl files in the given directories."""
    files = []
    for directory in input_dirs:
        path = Path(directory)
        if not path.is_dir():
            print(f"Warning: Input path is not a directory, skipping: {path}")
            continue
        
        dialog_file = path / "tau2_dialogs.jsonl"
        if dialog_file.exists():
            files.append(dialog_file)
        else:
            print(f"Warning: No 'tau2_dialogs.jsonl' found in {path}, skipping.")
    return files

def format_record(record: dict, stats: dict, run_name: str) -> dict | None:
    """
    Formats a single record from tau2_dialogs.jsonl into the format
    expected by the cookbook-internal workflow, while tracking stats.

    Args:
        record: The input JSON record from the dialogs file.
        stats: A dictionary for tracking processing statistics.
        run_name: The identifier for the run, derived from the input directory name.

    Returns:
        A formatted dictionary for the output JSONL, or None if the record is skipped.
    """
    stats['total_records_read'] += 1
    
    # Check for infrastructure failures first
    if record.get("metrics", {}).get("infra_failure", False):
        stats['infra_failures_skipped'] += 1
        return None

    try:
        domain = record["domain"]
        task_id = record["task_id"]
        group_id = f"{run_name}:{domain}:{task_id}"
        score = record.get("metrics", {}).get("score")
        
        # Explicitly check for None score, which indicates a legitimate but unscored run
        if score is None:
            stats['scoreless_records_skipped'] += 1
            return None
            
        stats['records_processed'] += 1
        return {
            "group_id": group_id,
            "domain_task": f"tau2:{domain}:{task_id}",
            "run_name": run_name,
            "messages": record["messages"],
            "exact_match": float(score),
            "original_session_id": record["session_id"],
            "original_task_id": task_id,
        }
    except (KeyError, TypeError) as e:
        stats['malformed_records_skipped'] += 1
        print(f"Warning: Skipping malformed record due to {e}: {record.get('session_id', 'Unknown session')}")
        return None

def get_task_split(args, all_task_ids: set) -> tuple[set, set]:
    """Determines the train/test split from either a manifest or a fraction."""
    if args.split_manifest:
        print(f"Loading split from manifest: {args.split_manifest}")
        with open(args.split_manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        train_task_ids = set(manifest["train_task_ids"])
        test_task_ids = set(manifest["test_task_ids"])
        # Ensure all tasks from the manifest are present in the current dataset
        if not (train_task_ids.union(test_task_ids)).issubset(all_task_ids):
            print("Warning: The provided manifest contains task IDs not found in the input data.")
        return train_task_ids, test_task_ids
    else:
        print(f"Creating new split with test fraction: {args.test_fraction}")
        task_id_list = sorted(list(all_task_ids), key=int)
        split_index = int(len(task_id_list) * args.test_fraction)
        test_task_ids = set(task_id_list[:split_index])
        train_task_ids = set(task_id_list[split_index:])
        return train_task_ids, test_task_ids

def main():
    """Main execution function."""
    args = parse_args()
    
    run_name_map = {}
    if args.run_name_map:
        try:
            for item in args.run_name_map:
                key, value = item.split(":", 1)
                run_name_map[key] = value
            print(f"Using run name mappings: {run_name_map}")
        except ValueError as e:
            print(f"Error: Invalid format for --run-name-map. Use 'dirname:run_name'. Details: {e}")
            return

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {output_path}")

    dialog_files = find_dialog_files(args.input_dirs)
    if not dialog_files:
        print("Error: No dialog files found. Exiting.")
        return

    print(f"Found {len(dialog_files)} dialog files to process.")

    # 1. First pass: Collect all unique task IDs
    all_task_ids = set()
    for file_path in dialog_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    all_task_ids.add(record["task_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    
    # 2. Get the train/test split for task IDs
    train_task_ids, test_task_ids = get_task_split(args, all_task_ids)
    
    print(f"Total unique tasks in data: {len(all_task_ids)}")
    print(f"  - Tasks in train split: {len(train_task_ids)}")
    print(f"  - Tasks in test split:  {len(test_task_ids)}")

    # 3. Save the definitive split manifest for this run
    split_manifest = {
        "train_task_ids": sorted(list(train_task_ids)),
        "test_task_ids": sorted(list(test_task_ids)),
        "source_manifest": args.split_manifest,
        "test_fraction_used": args.test_fraction,
    }
    manifest_path = output_path / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)
    print(f"Split manifest for this run saved to: {manifest_path}")

    # 4. Second pass: Process and write records
    train_file = output_path / "train.jsonl"
    test_file = output_path / "test.jsonl"
    
    # Initialize statistics counter
    stats = {
        "total_records_read": 0,
        "infra_failures_skipped": 0,
        "scoreless_records_skipped": 0,
        "malformed_records_skipped": 0,
        "records_processed": 0,
        "train_records_written": 0,
        "test_records_written": 0,
    }
    
    with open(train_file, "w", encoding="utf-8") as f_train, \
         open(test_file, "w", encoding="utf-8") as f_test:
        
        for file_path in dialog_files:
            dir_basename = file_path.parent.name
            run_name = run_name_map.get(dir_basename, dir_basename)
            with open(file_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    try:
                        record = json.loads(line)
                        task_id = record["task_id"]
                        
                        formatted = format_record(record, stats, run_name)
                        if not formatted:
                            continue

                        if task_id in test_task_ids:
                            f_test.write(json.dumps(formatted) + "\n")
                            stats['test_records_written'] += 1
                        elif task_id in train_task_ids:
                            f_train.write(json.dumps(formatted) + "\n")
                            stats['train_records_written'] += 1
                    except (json.JSONDecodeError, KeyError):
                        # This will now be caught by format_record, but as a fallback:
                        stats['malformed_records_skipped'] += 1
                        continue

    print("\n" + "="*70)
    print("DATA PROCESSING SUMMARY")
    print("="*70)
    print(f"Total records read from input files: {stats['total_records_read']}")
    print("-" * 20)
    print(f"Records skipped (Infrastructure Failure): {stats['infra_failures_skipped']}")
    print(f"Records skipped (Legitimate run, but no score): {stats['scoreless_records_skipped']}")
    print(f"Records skipped (Malformed JSON/structure): {stats['malformed_records_skipped']}")
    print("-" * 20)
    print(f"Total records successfully processed: {stats['records_processed']}")
    print(f"  - Wrote {stats['train_records_written']} records to {train_file}")
    print(f"  - Wrote {stats['test_records_written']} records to {test_file}")
    print("="*70)


if __name__ == "__main__":
    main()
