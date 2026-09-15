#!/usr/bin/env python3

import argparse
import random
import subprocess
import sys
import time

PROMPTS = [
    "What is 847 * 293?",
    "Calculate 12345 + 67890.",
    "What is 9876 divided by 12?",
    "Calculate 144 * 37.",
    "What is the current system status?",
    "Check the current system status and summarize it.",
    "Calculate 9999 - 4321.",
    "What is 256 * 64?",
    "Calculate 1728 / 24.",
    "What is 73 squared?",
    "Check the system status and explain whether everything looks healthy.",
    "Calculate 4567 + 8901.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--count", type=int, default=150)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    prompts = [rng.choice(PROMPTS) for _ in range(args.count)]

    print(f"Generating {args.count} baseline trajectories...")

    failures = 0

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{args.count}] {prompt}")

        result = subprocess.run(
            [sys.executable, "main.py", prompt],
            stdin=subprocess.DEVNULL,
        )

        if result.returncode != 0:
            failures += 1
            print(f"Run failed with exit code {result.returncode}")

        if args.delay:
            time.sleep(args.delay)

    print(f"\nFinished: {args.count - failures}/{args.count} runs succeeded")
    print("Trajectories saved under data/trajectories/")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
