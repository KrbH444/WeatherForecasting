#!/usr/bin/env python3
"""
Setup script for Weather Forecasting System
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def create_virtual_environment():
    """Create virtual environment if it doesn't exist"""
    venv_path = Path("tf2env")
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "tf2env"], check=True)
        print("✅ Virtual environment created successfully")
    else:
        print("✅ Virtual environment already exists")

def install_dependencies():
    """Install required dependencies"""
    print("📥 Installing dependencies...")
    
    # Determine the pip command based on OS
    if os.name == 'nt':  # Windows
        pip_cmd = "tf2env\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        pip_cmd = "tf2env/bin/pip"
    
    try:
        subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    if not env_file.exists():
        print("🔧 Creating .env file...")
        env_content = """# Flask Configuration
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-here

# Database Configuration
DATABASE_URL=sqlite:///weather.db

# API Configuration
OPENWEATHER_API_KEY=your-api-key-here
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ .env file created")
        print("⚠️  Please update the API keys in .env file")
    else:
        print("✅ .env file already exists")

def create_directories():
    """Create necessary directories"""
    directories = ["instance", "logs", "data", "models"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Directories created")

def setup_tailwind():
    """Setup Tailwind CSS"""
    print("🎨 Setting up Tailwind CSS...")
    
    # Check if Node.js is installed
    try:
        subprocess.run(["node", "--version"], check=True, capture_output=True)
        print("✅ Node.js detected")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Node.js not found. Please install Node.js to use Tailwind CSS")
        print("   Download from: https://nodejs.org/")
        return False
    
    # Install npm dependencies
    try:
        subprocess.run(["npm", "install"], check=True)
        print("✅ NPM dependencies installed")
        
        # Build CSS
        subprocess.run(["npm", "run", "build:css:prod"], check=True)
        print("✅ Tailwind CSS built successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error setting up Tailwind CSS: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Weather Forecasting System...")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Create virtual environment
    create_virtual_environment()
    
    # Install dependencies
    install_dependencies()
    
    # Create .env file
    create_env_file()
    
    # Create directories
    create_directories()
    
    # Setup Tailwind CSS
    setup_tailwind()
    
    print("=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Activate virtual environment:")
    if os.name == 'nt':  # Windows
        print("   tf2env\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print("   source tf2env/bin/activate")
    print("2. Update API keys in .env file")
    print("3. Initialize database: flask db upgrade")
    print("4. Run the application: python app.py")
    print("\n🌐 Access the application at: http://localhost:5000")

if __name__ == "__main__":
    main() 