#!/usr/bin/env python3
"""
Tailwind CSS Build Script for Weather Forecasting System
"""

import subprocess
import sys
import os
from pathlib import Path

def check_node_installed():
    """Check if Node.js is installed"""
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js version: {result.stdout.strip()}")
            return True
        else:
            print("❌ Node.js not found")
            return False
    except FileNotFoundError:
        print("❌ Node.js not found. Please install Node.js from https://nodejs.org/")
        return False

def install_dependencies():
    """Install npm dependencies"""
    print("📦 Installing npm dependencies...")
    try:
        subprocess.run(["npm", "install"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def build_css(production=False):
    """Build Tailwind CSS"""
    print("🎨 Building Tailwind CSS...")
    
    command = ["npm", "run", "build:css:prod" if production else "build:css"]
    
    try:
        subprocess.run(command, check=True)
        print("✅ CSS built successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error building CSS: {e}")
        return False

def watch_css():
    """Watch for CSS changes"""
    print("👀 Starting CSS watcher...")
    print("Press Ctrl+C to stop")
    
    try:
        subprocess.run(["npm", "run", "build:css"], check=True)
    except KeyboardInterrupt:
        print("\n👋 CSS watcher stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in CSS watcher: {e}")

def main():
    """Main function"""
    print("🎨 Tailwind CSS Build Script")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("package.json").exists():
        print("❌ package.json not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Check Node.js installation
    if not check_node_installed():
        sys.exit(1)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "install":
            install_dependencies()
        elif command == "build":
            build_css(production=True)
        elif command == "watch":
            watch_css()
        elif command == "dev":
            install_dependencies()
            build_css(production=False)
        else:
            print("Usage: python build_css.py [install|build|watch|dev]")
            print("  install - Install npm dependencies")
            print("  build   - Build production CSS")
            print("  watch   - Watch for changes and rebuild")
            print("  dev     - Install dependencies and build development CSS")
    else:
        # Default: install and build
        print("🔧 Setting up Tailwind CSS...")
        if install_dependencies():
            build_css(production=True)

if __name__ == "__main__":
    main() 