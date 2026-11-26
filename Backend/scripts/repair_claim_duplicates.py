"""Dry-run or apply conservative claim relationship repair for one paper."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pymongo.errors import PyMongoError
from app.db.mongodb import close_mongo_client, get_database
from app.modules.claim_extraction.repair import repair_paper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        result = repair_paper(get_database(), args.paper_id, args.apply)
        print(f"Examined: {result['examined']}; planned: {result['planned']}; applied: {result['applied']}")
        for change in result["changes"]:
            action = "requires re-verification" if change["requires_reverification"] else "deterministic exact match"
            print(f"{change['claim_id']}: {action}; old target={change['previous_target']}; new target={change['new'].get('duplicate_of_claim_id')}")
    except (PyMongoError, RuntimeError):
        raise SystemExit("Database operation failed. Check the connection configuration.")
    finally:
        close_mongo_client()


if __name__ == "__main__":
    main()
