#!/usr/bin/env python
"""
Quick Start Script for Bloque2_CNN Training & Evaluation
Usage: python quick_start.py [command] [task] [model]
"""

import sys
import subprocess
import argparse
from pathlib import Path
import shutil

# Colors for terminal output (Windows + Unix)
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
    print(f"{msg}")
    print(f"{'='*60}{Colors.RESET}\n")

class QuickStart:
    def __init__(self):
        self.config_file = "config.yaml"
        self.root_dir = Path(__file__).parent
        
    def check_python(self):
        """Verify Python version"""
        version = f"Python {sys.version.split()[0]}"
        print_success(f"{version}")
        
        if sys.version_info < (3, 8):
            print_error("Python 3.8+ required")
            return False
        return True
    
    def check_pytorch(self):
        """Check PyTorch installation and CUDA availability"""
        try:
            import torch
            print_success(f"PyTorch {torch.__version__}")
            
            if torch.cuda.is_available():
                print_success(f"CUDA available - GPU: {torch.cuda.get_device_name(0)}")
                print_success(f"CUDA Version: {torch.version.cuda}")
            else:
                print_warning("CUDA not available - will use CPU (slow)")
            return True
        except ImportError:
            print_error("PyTorch not installed")
            return False
    
    def install_dependencies(self):
        """Install requirements"""
        print_header("Installing Dependencies")
        requirements_file = self.root_dir / "requirements.txt"
        
        if not requirements_file.exists():
            print_error(f"requirements.txt not found at {requirements_file}")
            return False
        
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file), "-q"],
                check=True,
                cwd=str(self.root_dir)
            )
            print_success("Dependencies installed")
            return True
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install dependencies: {e}")
            return False
    
    def train(self, task="all", model="unet"):
        """Train models"""
        print_header(f"Training - Task: {task}, Model: {model}")
        
        try:
            cmd = [
                sys.executable, "-m", "src.train",
                "--config", self.config_file,
                "--task", task,
                "--model", model
            ]
            
            result = subprocess.run(
                cmd,
                cwd=str(self.root_dir),
                capture_output=False
            )
            
            if result.returncode == 0:
                print_success("Training completed")
                return True
            else:
                print_error("Training failed")
                return False
        except Exception as e:
            print_error(f"Training error: {e}")
            return False
    
    def evaluate(self, task="all", seg_model="unet"):
        """Evaluate models"""
        print_header(f"Evaluation - Task: {task}, Model: {seg_model}")
        
        try:
            cmd = [
                sys.executable, "-m", "src.evaluate",
                "--config", self.config_file,
                "--task", task,
                "--seg-model", seg_model
            ]
            
            result = subprocess.run(
                cmd,
                cwd=str(self.root_dir),
                capture_output=False
            )
            
            if result.returncode == 0:
                print_success("Evaluation completed")
                return True
            else:
                print_error("Evaluation failed")
                return False
        except Exception as e:
            print_error(f"Evaluation error: {e}")
            return False
    
    def setup(self):
        """Check if environment is ready"""
        print_header("Checking Environment")
        
        checks = [
            ("Python", self.check_python),
            ("PyTorch", self.check_pytorch),
        ]
        
        all_ok = True
        for name, check_func in checks:
            print(f"Checking {name}...", end=" ")
            sys.stdout.flush()
            try:
                if not check_func():
                    all_ok = False
            except Exception as e:
                print_error(f"{name} check failed: {e}")
                all_ok = False
        
        if all_ok:
            print_success("Environment ready for training")
        else:
            print_error("Environment check failed")
        
        return all_ok

def main():
    parser = argparse.ArgumentParser(
        description="Quick Start for Bloque2_CNN Training & Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_start.py setup
  python quick_start.py install
  python quick_start.py train --task classification
  python quick_start.py eval --task detection
  python quick_start.py train --task segmentation --model unet
  python quick_start.py full --task all
        """
    )
    
    parser.add_argument(
        "command",
        choices=["setup", "install", "train", "eval", "full"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--task",
        default="all",
        choices=["all", "classification", "detection", "segmentation"],
        help="Task to run (default: all)"
    )
    
    parser.add_argument(
        "--model",
        default="unet",
        choices=["unet", "deeplabv3plus"],
        help="Segmentation model (default: unet)"
    )
    
    args = parser.parse_args()
    
    qs = QuickStart()
    
    print_header("Bloque2_CNN Training & Evaluation Quick Start")
    
    # Execute command
    if args.command == "setup":
        success = qs.setup()
    
    elif args.command == "install":
        success = qs.install_dependencies()
    
    elif args.command == "train":
        if not qs.setup():
            print_error("Environment check failed. Run 'python quick_start.py install' first.")
            return 1
        success = qs.train(args.task, args.model)
    
    elif args.command == "eval":
        if not qs.setup():
            print_error("Environment check failed. Run 'python quick_start.py install' first.")
            return 1
        success = qs.evaluate(args.task, args.model)
    
    elif args.command == "full":
        if not qs.setup():
            print_error("Environment check failed. Run 'python quick_start.py install' first.")
            return 1
        success = qs.train(args.task, args.model) and qs.evaluate(args.task, args.model)
    
    print()
    if success or args.command in ["setup", "install"]:
        print_success("Done!")
        return 0
    else:
        print_error("Failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
