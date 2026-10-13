import subprocess, sys, pathlib, argparse, shlex

ROOT = pathlib.Path(__file__).resolve.parent
OUT = ROOT / "out"

def run(cmd: str):
    print(f"\n>>> {cmd}")

    res = subprocess.run(
        shlex.split(cmd),
        cwd=ROOT
    )

    if res.returncode != 0:
        sys.exit(res.returncode)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--visulize", action="store_true", help="run visulization scrips and save PNGs to ./out")
    args = p.parse_args()

    OUT.mkdir(exist_ok=True)

    # NumPy tiny example
    run("python attn_numpy_demo.py")

    # 1.3 and 1.4 Unit tests
    run("python -m pytest -q tests/test_attn_math.py")
    run("python -m pytest -q tests/test_casual_mask.py")

    # Matrix math walkthrough
    run("python demo_mha_shapes.py")

    if args.vizualize:
        run("python demo_vizualize_multi_head.py")
        print("\nAll Part 1 demo/tests completed.")

if __name__ == "__main__":
    main()