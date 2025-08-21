#!/usr/bin/env python
# Simple CLI script to generate embeddings for Bible verses

import os
import sys
import subprocess

def main():
    """Entry point for the embedding generation CLI"""
    print("📚 Bible Embedding Generator")
    print("=" * 50)
    
    # Get Python executable path from the environment
    python_path = sys.executable
    
    # Define options
    options = [
        ("Generate embeddings for all Bible versions", ""),
        ("Generate embeddings for KJV only", "--version kjv"),
        ("Generate embeddings for NIV only", "--version niv"),
        ("Generate embeddings for NKJV only", "--version nkjv"),
        ("Generate embeddings for NLT only", "--version nlt"),
        ("Force regenerate all embeddings (all versions)", "--force"),
        ("Exit", None)
    ]
    
    # Display menu
    print("Choose an option:")
    for i, (label, _) in enumerate(options, start=1):
        print(f"{i}. {label}")
    
    # Get user choice
    choice = int(input("\nEnter your choice (1-7): ").strip())
    
    # Exit if requested
    if choice == 7:
        print("Exiting...")
        return
    
    # Validate choice
    if choice < 1 or choice > len(options):
        print(f"Invalid choice: {choice}")
        return
    
    # Get the selected option
    option_label, option_args = options[choice-1]
    
    print(f"\n🚀 {option_label}")
    
    # Build the command
    cmd = [python_path, "-m", "utils.generate_embeddings"]
    if option_args:
        cmd.extend(option_args.split())
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)
    
    # Execute the command
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Embedding generation completed!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running command: {e}")
    
if __name__ == "__main__":
    main()
