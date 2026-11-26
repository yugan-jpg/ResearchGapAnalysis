"""Preview or apply source-checked resolution of verified duplicate groups."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pymongo.errors import PyMongoError
from app.db.mongodb import close_mongo_client, get_database
from app.modules.claim_extraction.resolution import resolve_paper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        result = resolve_paper(get_database(), args.paper_id, apply=args.apply)
        print(f"Examined: {result['examined']}; planned: {result['planned']}; applied: {result['applied']}")
        for change in result["changes"]:
            print(f"{change['claim_id']}: canonical={change['canonical_id']}; status={change['new']['verification_status']}")
    except (PyMongoError, RuntimeError):
        raise SystemExit("Database operation failed. Check the connection configuration.")
    finally:
        close_mongo_client()


if __name__ == "__main__":
    main()
