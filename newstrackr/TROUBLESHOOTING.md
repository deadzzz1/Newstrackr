# 🔧 Troubleshooting Guide - Newstrackr

Common issues and solutions for the Newstrackr application.

## 🚨 Quick Fix: Missing Dependencies

### Error: `ModuleNotFoundError: No module named 'newsapi'`

**Solution:**
```bash
pip install newsapi-python
```

**Alternative (Windows):**
```cmd
python install_deps.py
```

### Error: `ModuleNotFoundError: No module named 'transformers'`

**Solution:**
```bash
pip install transformers torch
```

**Note:** These packages are large (~2GB). The app will work with basic summarization without them.

### Error: `ModuleNotFoundError: No module named 'sklearn'`

**Solution:**
```bash
pip install scikit-learn
```

## 🐍 Python Environment Issues

### Issue: `python: command not found` (Linux/Mac)

**Solution:**
```bash
# Try python3 instead
python3 -m pip install -r requirements.txt
python3 app.py
```

### Issue: `externally-managed-environment` (Linux)

**Solution 1 - Virtual Environment (Recommended):**
```bash
python3 -m venv newstrackr_env
source newstrackr_env/bin/activate  # Linux/Mac
# or
newstrackr_env\Scripts\activate     # Windows
pip install -r requirements.txt
```

**Solution 2 - User Install:**
```bash
pip install --user -r requirements.txt
```

## 🖥️ Streamlit Issues

### Issue: `streamlit: command not found`

**Solution:**
```bash
# Install streamlit
pip install streamlit

# Or run directly with python
python -m streamlit run app.py
```

### Issue: Port already in use

**Solution:**
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

### Issue: Browser doesn't open automatically

**Solution:**
Manually open: `http://localhost:8501` in your browser

## 🤖 Model Issues

### Issue: Impact ranking model not found

**Error:** `FileNotFoundError: models/impact_ranker.pkl`

**Solution:**
```bash
python train_model.py
```

### Issue: BART model download fails

**Symptoms:**
- Very slow loading
- Connection timeouts
- Memory errors

**Solutions:**
1. **Use basic summarization:**
   - The app will automatically fall back to basic summarization
   - No action needed

2. **For full AI features (if you have good internet/RAM):**
   ```bash
   pip install transformers torch
   ```

## 📊 Data Issues

### Issue: No news articles displayed

**Possible Causes:**
1. **No API key:** App uses sample data (this is normal)
2. **Invalid API key:** Check your NewsAPI key
3. **API quota exceeded:** NewsAPI free tier has limits
4. **Network issues:** Check internet connection

**Solutions:**
- The app works with sample data
- Get a free API key from [newsapi.org](https://newsapi.org/)
- Check API quota on NewsAPI dashboard

### Issue: "Error fetching news"

**Solution:**
1. Check internet connection
2. Verify API key is valid
3. App will fall back to sample data

## 💾 Memory Issues

### Issue: "Killed" or memory errors

**Cause:** BART model requires ~2GB RAM

**Solutions:**
1. **Skip AI packages:**
   ```bash
   # Install without transformers/torch
   pip install streamlit pandas numpy requests plotly scikit-learn newsapi-python
   ```

2. **Use basic summarization:**
   - App automatically detects missing packages
   - Provides simple text summarization instead

3. **Increase system memory:**
   - Close other applications
   - Consider cloud deployment

## 🌐 Windows-Specific Issues

### Issue: Package installation fails on Windows

**Solution 1 - Use our installer:**
```cmd
python install_deps.py
```

**Solution 2 - Update pip:**
```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Solution 3 - Use conda (if available):**
```cmd
conda install streamlit pandas numpy scikit-learn
pip install newsapi-python transformers torch
```

### Issue: PowerShell execution policy

**Error:** `cannot be loaded because running scripts is disabled`

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 🔄 Setup Scripts

### Option 1: Automated installer (Windows)
```cmd
python install_deps.py
```

### Option 2: Batch file (Windows)
```cmd
setup_windows.bat
```

### Option 3: Manual setup
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## 📱 Browser Issues

### Issue: App doesn't load in browser

**Solutions:**
1. **Try different browser:** Chrome, Firefox, Edge
2. **Clear browser cache**
3. **Disable browser extensions**
4. **Check console for errors** (F12 in most browsers)

### Issue: App looks broken/unstyled

**Solutions:**
1. **Hard refresh:** Ctrl+F5 (Windows) or Cmd+Shift+R (Mac)
2. **Check browser compatibility:** Use modern browser
3. **Disable ad blockers** temporarily

## ⚡ Performance Issues

### Issue: App is slow to load

**Causes & Solutions:**
1. **First-time model download:**
   - BART model is large (~1.6GB)
   - Subsequent loads will be faster
   
2. **Limited system resources:**
   - Close other applications
   - Use basic summarization mode
   
3. **API rate limits:**
   - NewsAPI has request limits
   - App automatically caches data

### Issue: Summarization is slow

**Solutions:**
1. **Use basic summarization:**
   - Uninstall transformers: `pip uninstall transformers torch`
   - App will use faster basic method
   
2. **Reduce article length:**
   - Summarizer works best with shorter texts
   
3. **Check system resources:**
   - BART requires significant CPU/RAM

## 🆘 Getting Help

### Still having issues?

1. **Check the error message carefully**
2. **Look for similar issues in this guide**
3. **Try the demo script:**
   ```bash
   python demo.py
   ```
4. **Check system requirements:**
   - Python 3.7+
   - 4GB+ RAM (for AI features)
   - Internet connection

### Minimal Installation

If all else fails, try the minimal installation:

```bash
# Core packages only
pip install streamlit pandas numpy requests plotly

# Run with basic features
streamlit run app.py
```

### Environment Information

To help with debugging, check:
```bash
python --version
pip --version
pip list | grep -E "(streamlit|pandas|numpy|transformers)"
```

## 📞 Support

- **Documentation:** README.md
- **Demo:** `python demo.py`
- **Quick Start:** QUICKSTART.md
- **Issues:** Create GitHub issue with error details

---

**Most issues are dependency-related and can be fixed with:** `python install_deps.py` 🔧