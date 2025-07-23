#!/usr/bin/env python3
"""
Setup script for Newstrackr
Installs dependencies and trains the impact ranking model
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies"""
    print("🔧 Installing dependencies...")
    
    try:
        # Try to install with pip
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--user",
            "streamlit", "pandas", "numpy", "requests", "transformers", 
            "torch", "scikit-learn", "plotly", "pycountry", "newsapi-python",
            "beautifulsoup4", "tweepy", "python-dotenv", "joblib", "nltk",
            "wordcloud", "seaborn", "matplotlib"
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies with pip")
        print("Please install dependencies manually:")
        print("pip install -r requirements.txt")
        return False

def train_model():
    """Train the impact ranking model"""
    print("\n🤖 Training impact ranking model...")
    
    try:
        # Import and run training
        from train_model import train_model as run_training
        run_training()
        print("✅ Model training completed!")
        return True
    except Exception as e:
        print(f"❌ Model training failed: {str(e)}")
        print("You can train the model manually by running:")
        print("python3 train_model.py")
        return False

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    
    directories = ['models', 'data', 'notebooks']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    """Main setup function"""
    print("🚀 Setting up Newstrackr - News Aggregator & Summarizer")
    print("=" * 60)
    
    # Setup directories
    setup_directories()
    
    # Install dependencies
    deps_success = install_dependencies()
    
    if deps_success:
        # Train model
        model_success = train_model()
        
        if model_success:
            print("\n🎉 Setup completed successfully!")
            print("\n📋 Next steps:")
            print("1. (Optional) Get a NewsAPI key from https://newsapi.org/")
            print("2. Run the app: streamlit run app.py")
            print("3. Enter your NewsAPI key in the sidebar (or use sample data)")
            print("4. Enjoy exploring news with AI-powered insights!")
        else:
            print("\n⚠️  Setup completed with warnings.")
            print("The app will work, but you may need to train the model manually.")
    else:
        print("\n❌ Setup failed. Please install dependencies manually and try again.")

if __name__ == "__main__":
    main()