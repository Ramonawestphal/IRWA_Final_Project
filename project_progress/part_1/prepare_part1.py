import argparse, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # add project root

from preprocess import preprocess_jsonl

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    a = ap.parse_args()
    preprocess_jsonl(a.inp, a.out)