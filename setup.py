#!/usr/bin/env python3
"""
Setup Script for AI Lead Scoring Dashboard
==========================================
This script automates the complete setup process for the lead scoring system.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}:")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return None

def check_file_exists(filename):
    """Check if a file exists."""
    return Path(filename).exists()

def main():
    """Main setup function."""
    print("🎯 AI Lead Scoring Dashboard Setup")
    print("=" * 50)
    
    # Check if required files exist
    required_files = [
        "app.py",
        "train_model.py", 
        "generate_sample_data.py",
        "requirements.txt"
    ]
    
    print("\n📋 Checking required files...")
    missing_files = []
    for file in required_files:
        if check_file_exists(file):
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} missing")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n🚫 Missing files: {missing_files}")
        print("Please ensure all required files are in the current directory.")
        sys.exit(1)
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    install_result = run_command("pip install -r requirements.txt", "Installing Python packages")
    if install_result is None:
        print("Failed to install dependencies. Please run 'pip install -r requirements.txt' manually.")
        sys.exit(1)
    
    # Generate sample data if it doesn't exist
    if not check_file_exists("leads.csv"):
        print("\n📊 Generating sample dataset...")
        data_result = run_command("python generate_sample_data.py", "Generating sample leads data")
        if data_result is None:
            print("Failed to generate sample data.")
            sys.exit(1)
    else:
        print("\n✅ Sample dataset already exists (leads.csv)")
    
    # Train the model if it doesn't exist
    if not check_file_exists("model.pkl"):
        print("\n🤖 Training machine learning model...")
        train_result = run_command("python train_model.py", "Training the lead scoring model")
        if train_result is None:
            print("Failed to train the model.")
            sys.exit(1)
    else:
        print("\n✅ Trained model already exists (model.pkl)")
    
    # Final verification
    print("\n🔍 Final verification...")
    final_files = ["leads.csv", "model.pkl"]
    all_good = True
    
    for file in final_files:
        if check_file_exists(file):
            print(f"✅ {file} ready")
        else:
            print(f"❌ {file} missing")
            all_good = False
    
    if all_good:
        print("\n🎉 Setup completed successfully!")
        print("\n🚀 To start the dashboard, run:")
        print("   streamlit run app.py")
        print("\n🌐 The dashboard will be available at: http://localhost:8501")
        print("\n📚 For deployment instructions, see README.md")
        
        # Ask if user wants to start the app
        try:
            start_app = input("\nStart the dashboard now? (y/n): ").lower().strip()
            if start_app in ['y', 'yes']:
                print("\n🚀 Starting the dashboard...")
                subprocess.run("streamlit run app.py", shell=True)
        except KeyboardInterrupt:
            print("\n\n👋 Setup completed. Run 'streamlit run app.py' when ready!")
    else:
        print("\n❌ Setup incomplete. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()