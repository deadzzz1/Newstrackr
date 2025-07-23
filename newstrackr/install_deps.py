#!/usr/bin/env python3
"""
Simple dependency installer for Newstrackr
Handles Windows-specific installation issues
"""

import subprocess
import sys
import os

def install_package(package_name, display_name=None):
    """Install a single package with error handling"""
    if display_name is None:
        display_name = package_name
    
    print(f"📦 Installing {display_name}...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", package_name
        ])
        print(f"✅ {display_name} installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {display_name}: {e}")
        return False

def install_core_dependencies():
    """Install core dependencies needed to run the app"""
    print("🔧 Installing core dependencies for Newstrackr...")
    print("=" * 50)
    
    # Core packages needed for basic functionality
    core_packages = [
        ("streamlit", "Streamlit"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("requests", "Requests"),
        ("plotly", "Plotly"),
        ("scikit-learn", "Scikit-learn"),
        ("joblib", "Joblib"),
        ("pycountry", "PyCountry"),
        ("python-dotenv", "Python-dotenv"),
        ("newsapi-python", "NewsAPI Python"),
        ("beautifulsoup4", "BeautifulSoup4"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
    ]
    
    # Optional ML packages (heavy)
    ml_packages = [
        ("transformers", "Transformers (for AI summarization)"),
        ("torch", "PyTorch (for AI models)"),
    ]
    
    success_count = 0
    total_count = len(core_packages)
    
    # Install core packages
    for package, display_name in core_packages:
        if install_package(package, display_name):
            success_count += 1
    
    print(f"\n📊 Core Installation Summary: {success_count}/{total_count} packages installed")
    
    # Ask about ML packages
    if success_count == total_count:
        print("\n🤖 Optional AI packages (large downloads ~2GB):")
        install_ml = input("Install AI summarization packages? (y/N): ").lower().startswith('y')
        
        if install_ml:
            ml_success = 0
            for package, display_name in ml_packages:
                if install_package(package, display_name):
                    ml_success += 1
            
            if ml_success == len(ml_packages):
                print("✅ AI packages installed! Full functionality available.")
            else:
                print("⚠️ Some AI packages failed. Basic summarization will be used.")
        else:
            print("ℹ️ Skipping AI packages. Basic summarization will be used.")
    
    return success_count >= (total_count - 2)  # Allow 2 failures

def train_model_if_needed():
    """Train the impact model if not present"""
    model_path = os.path.join("models", "impact_ranker.pkl")
    
    if not os.path.exists(model_path):
        print("\n🤖 Training impact ranking model...")
        try:
            # Import here to avoid circular imports
            from train_model import train_model
            train_model()
            print("✅ Model training completed!")
            return True
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            print("💡 You can train manually later with: python train_model.py")
            return False
    else:
        print("✅ Impact ranking model already exists!")
        return True

def main():
    """Main installation function"""
    print("🚀 Newstrackr Dependency Installer")
    print("=" * 50)
    print("This will install the required packages for Newstrackr to run.")
    print("Note: Some packages are large and may take time to download.\n")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ is required. You have:", sys.version)
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Install dependencies
    deps_success = install_core_dependencies()
    
    if deps_success:
        print("\n📁 Setting up directories...")
        os.makedirs("models", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        # Train model
        model_success = train_model_if_needed()
        
        print("\n" + "=" * 50)
        print("🎉 Installation Summary")
        print("=" * 50)
        
        if model_success:
            print("✅ All components installed successfully!")
            print("\n📋 Next steps:")
            print("1. Run the app: streamlit run app.py")
            print("2. (Optional) Get NewsAPI key from https://newsapi.org/")
            print("3. Enjoy exploring news with AI!")
        else:
            print("⚠️ Installation completed with warnings.")
            print("The app will work, but some features may be limited.")
            
        return True
    else:
        print("\n❌ Installation failed.")
        print("Try installing manually with:")
        print("pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🚀 Ready to run: streamlit run app.py")
        
        input("\nPress Enter to exit...")
    except KeyboardInterrupt:
        print("\n\n❌ Installation cancelled by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please try manual installation with: pip install -r requirements.txt")