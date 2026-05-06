#!/usr/bin/env python
"""
Quick Start Script for Bloque2_CNN Training & Evaluation
"""

import sys
import subprocess
import argparse
from pathlib import Path


# ===== COLORS =====
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'


def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")


def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.RESET}")


def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.RESET}")


def print_header(msg):
    print(f"\n{Colors.GREEN}{'='*60}")
    print(msg)
    print(f"{'='*60}{Colors.RESET}\n")


# ===== MAIN CLASS =====
class QuickStart:
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.config_file = self.root_dir / "config.yaml"

    # ===== CHECKS =====
    def check_python(self):
        version = sys.version_info
        print_success(f"Python {sys.version.split()[0]}")

        if version < (3, 8) or version >= (3, 13):
            print_error("Use Python 3.8 - 3.12 (PyTorch compatible)")
            return False
        return True

    def check_config(self):
        if not self.config_file.exists():
            print_error(f"Config file not found: {self.config_file}")
            return False
        print_success("Config file found")
        return True

    def check_pytorch(self):
        try:
            import torch
            print_success(f"PyTorch {torch.__version__}")

            if torch.cuda.is_available():
                print_success(f"CUDA OK - {torch.cuda.get_device_name(0)}")
                print_success(f"CUDA Runtime: {torch.version.cuda}")
            else:
                print_warning("CUDA NOT available → using CPU")

            return True

        except ImportError:
            print_warning("PyTorch not installed")
            return False

    # ===== INSTALL =====
    def install_dependencies(self):
        print_header("Installing Dependencies")

        req_file = self.root_dir / "requirements.txt"

        if not req_file.exists():
            print_error("requirements.txt not found")
            return False

        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
                check=True
            )
            print_success("Dependencies installed")
            print_warning("PyTorch is handled separately in your conda environment")
            return True

        except subprocess.CalledProcessError as e:
            print_error(e)
            return False

    def install_pytorch(self):
        print_header("Installing PyTorch (CUDA)")

        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ], check=True)

            print_success("PyTorch CUDA installed")
            return True

        except subprocess.CalledProcessError as e:
            print_error(e)
            return False

    # ===== TRAIN / EVAL =====
    def run_command(self, module, extra_args):
        cmd = [sys.executable, "-m", module] + extra_args

        print_warning(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=str(self.root_dir),
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print_error("Execution failed:")
            print(result.stderr)
            return False

        print(result.stdout)
        return True

    def train(self, task, model):
        print_header(f"TRAIN → {task} | {model}")
        return self.run_command("src.train", [
            "--config", str(self.config_file),
            "--task", task
        ])

    def evaluate(self, task, model):
        print_header(f"EVAL → {task} | {model}")
        return self.run_command("src.evaluate", [
            "--config", str(self.config_file),
            "--task", task,
            "--seg-model", model
        ])

    # ===== SETUP =====
    def setup(self):
        print_header("Environment Check")

        checks = [
            self.check_python,
            self.check_config,
            self.check_pytorch
        ]

        ok = True
        for c in checks:
            if not c():
                ok = False

        if ok:
            print_success("Environment READY")
        else:
            print_error("Environment NOT ready")

        return ok


# ===== CLI =====
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=["setup", "install", "train", "eval", "full"]
    )

    parser.add_argument(
        "--task",
        default="all",
        choices=["all", "classification", "detection", "segmentation"]
    )

    parser.add_argument(
        "--model",
        default="unet",
        choices=["unet", "deeplabv3plus"]
    )

    args = parser.parse_args()

    qs = QuickStart()

    print_header("Bloque2_CNN Quick Start")

    if args.command == "setup":
        qs.setup()

    elif args.command == "install":
        qs.install_dependencies()

    elif args.command == "train":
        if qs.setup():
            qs.train(args.task, args.model)

    elif args.command == "eval":
        if qs.setup():
            qs.evaluate(args.task, args.model)

    elif args.command == "full":
        if qs.setup():
            print_warning("Full pipeline may take HOURS")
            qs.train(args.task, args.model)
            qs.evaluate(args.task, args.model)


if __name__ == "__main__":
    sys.exit(main())