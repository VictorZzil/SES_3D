"""Recompute the paper's retained-score analyses and check the archived outputs."""

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile


EVIDENCE = Path(__file__).resolve().parent / "evidence"


def run(script, *args):
    result = subprocess.run([sys.executable, str(script), *map(str, args)],
                            capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{script.name} failed:\n{result.stdout}{result.stderr}")


def records(path):
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def check_outputs(folder, archive, names):
    for name in names:
        if records(folder / name) != records(archive / name):
            raise ValueError(f"Recomputed output differs from the archived result: {name}")


def main():
    with TemporaryDirectory(prefix="sesad-replay-") as folder:
        output = Path(folder)
        alarm = EVIDENCE / "alarm_readouts"
        inputs = (alarm / "fixed_split.csv", alarm / "readout_scores.csv")
        run(alarm / "sesad_alarm_readout_comparison.py", "replay", *inputs, output / "alarms")
        run(alarm / "sesad_matched_low_fpr.py", *inputs, output / "roc")
        alarm_names = ("readout_fixed_decisions.csv", "readout_fixed_thresholds.json",
                       "readout_repeat_calibration_groups.csv", "readout_repeat_metrics.csv",
                       "readout_protocol.json", "readout_summary.json")
        roc_names = ("matched_low_fpr.json", "matched_low_fpr_by_class.csv")
        check_outputs(output / "alarms", alarm, alarm_names)
        check_outputs(output / "roc", alarm, roc_names)
        print("Alarm thresholds, common false-alarm rates, and crop stress test: 8 files agree.")

        slots = EVIDENCE / "slot_reversals"
        slot_output = output / "rankings"
        slot_output.mkdir()
        with ZipFile(slots / "slot_scores.zip") as archive:
            for name in archive.namelist():
                if not (slot_output / name).resolve().is_relative_to(slot_output.resolve()):
                    raise ValueError(f"Unexpected path in score archive: {name}")
            archive.extractall(slot_output)
        run(slots / "sesad_slot_reversal_audit.py", "summarize", slot_output)
        check_outputs(slot_output, slots, ("slot_reversals.json",))
        scan_count = records(slot_output / "slot_reversals.json")["verified_accepted_scans"]
        print(f"Same-slot ranking audit: {scan_count:,} accepted scans agree.")

        paired = EVIDENCE / "fullbank_paired"
        paired_output = shutil.copytree(paired, output / "paired")
        run(paired_output / "figure_scripts" / "compute_fullbank_paired_v29.py")
        check_outputs(paired_output / "data", paired / "data",
                      ("fullbank_paired_v29.csv", "fullbank_paired_v29.json"))
        count = len(records(paired_output / "data" / "fullbank_paired_v29.csv")) - 1
        print(f"Paired full-bank bootstrap: {count} intervals agree.")


if __name__ == "__main__":
    main()
