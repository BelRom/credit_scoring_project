from __future__ import annotations
import sys
import great_expectations as ge


def run_checkpoint(checkpoint_name: str = "credit_default_checkpoint") -> int:
    context = ge.get_context()
    result = context.run_checkpoint(checkpoint_name=checkpoint_name)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "credit_default_checkpoint"
    raise SystemExit(run_checkpoint(checkpoint))
