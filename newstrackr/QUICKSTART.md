# 🚀 Quick Start Guide - Newstrackr

Get your Newstrackr app running in under 5 minutes!

## ⚡ Option 1: Automatic Setup (Recommended)

### For Windows:
```cmd
# 1. Navigate to the project
cd newstrackr

# 2. Run the dependency installer
python install_deps.py

# 3. Start the app
streamlit run app.py
```

### For Linux/Mac:
```bash
# 1. Navigate to the project
cd newstrackr

# 2. Run automatic setup
python3 setup.py

# 3. Start the app
streamlit run app.py
```

## 🛠️ Option 2: Manual Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Impact Model
```bash
python3 train_model.py
```

### Step 3: Run the App
```bash
streamlit run app.py
```

## 🔑 Getting a NewsAPI Key (Optional)

1. Visit [newsapi.org](https://newsapi.org/)
2. Sign up for a free account
3. Copy your API key
4. Enter it in the Streamlit sidebar

**Note:** The app works with sample data even without an API key!

## 🎯 First Steps in the App

1. **Explore Sample Data**: The app loads with sample news articles
2. **Try Filters**: Use the sidebar to filter by country, category, date
3. **Generate Summaries**: Click "Generate Summary" on any article
4. **View Analytics**: Check the dashboard metrics and charts
5. **Search**: Use the search bar to find specific topics

## 📊 Demo Features

Run the demo to see all features in action:
```bash
python3 demo.py
```

## 🔧 Troubleshooting

### Common Issues:

**"Model not found" error:**
```bash
python3 train_model.py
```

**"Dependencies missing" error:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**"Port already in use" error:**
```bash
streamlit run app.py --server.port 8502
```

## 📱 Using the App

### Sidebar Controls:
- **NewsAPI Key**: Enter for live data (optional)
- **Filters**: Country, category, date range, article count
- **Advanced**: Impact score and engagement thresholds

### Main Interface:
- **Dashboard**: Overview metrics and charts
- **Articles**: Expandable cards with full details
- **Search**: Real-time search across titles and descriptions

### Article Features:
- **Impact Score**: ML-generated relevance ranking
- **Social Metrics**: Simulated Twitter engagement
- **AI Summary**: One-click BART-generated summaries
- **Source Links**: Direct links to original articles

## 🎨 Customization

### Add Your Own Data:
- Replace sample data in `utils/fetch_live_news.py`
- Modify Twitter metrics simulation
- Train model with real engagement data

### Styling:
- Edit CSS in `app.py`
- Modify dashboard charts in `utils/widgets.py`
- Add new visualization components

## 📈 Performance Tips

1. **Caching**: News data cached for 1 hour
2. **Model Loading**: Models loaded once per session
3. **API Limits**: Free NewsAPI allows 1000 requests/day
4. **Memory**: BART model requires ~2GB RAM

## 🚀 Deployment

### Local Development:
```bash
streamlit run app.py
```

### Production:
- **Streamlit Cloud**: Connect GitHub repo
- **Heroku**: Use Streamlit buildpack
- **Docker**: See Dockerfile in README

## 📚 Learn More

- **Full Documentation**: See `README.md`
- **Model Training**: Check `notebooks/train_impact_ranker.ipynb`
- **Code Structure**: Explore `utils/` directory
- **Demo**: Run `python3 demo.py`

## 🆘 Need Help?

1. **Check README.md** for detailed documentation
2. **Run the demo** with `python3 demo.py`
3. **View logs** in the terminal for error details
4. **Create an issue** on GitHub for bugs or questions

---

**Ready to explore news with AI? Start with:** `streamlit run app.py` 🚀