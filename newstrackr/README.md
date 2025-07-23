# 📰 Newstrackr - AI-Powered News Aggregator & Summarizer

A sophisticated news aggregation and summarization application built with Streamlit, featuring ML-driven impact scoring and real-time summarization.

## 🚀 Features

### Core Functionality
- **Real-time News Fetching**: Integrates with NewsAPI for live news data
- **AI-Powered Summarization**: Uses facebook/bart-large-cnn for 2-paragraph summaries (~300 chars)
- **ML Impact Scoring**: Custom RandomForest model with TF-IDF and Twitter metrics
- **Smart Caching**: 1-day cache for optimal performance
- **Advanced Filtering**: Date range, country, category, and engagement filters

### User Interface
- **Modern Streamlit UI**: Clean, responsive design with custom CSS
- **Interactive Dashboards**: Metrics overview and data visualizations
- **Expandable Articles**: Click to reveal detailed information and summaries
- **Real-time Search**: Search across titles and descriptions
- **Social Media Metrics**: Simulated Twitter engagement data

### Machine Learning Models
1. **Summarizer**: facebook/bart-large-cnn transformer model
2. **Impact Ranker**: TF-IDF vectorizer + RandomForestRegressor

## 📁 Project Structure

```
newstrackr/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── models/
│   └── impact_ranker.pkl          # Trained impact ranking model
├── notebooks/
│   ├── eda.ipynb                  # Exploratory data analysis
│   ├── feature_engg.ipynb         # Feature engineering
│   └── train_impact_ranker.ipynb  # Model training notebook
├── utils/
│   ├── summarizer.py              # AI summarization utilities
│   ├── fetch_live_news.py         # News fetching and metrics
│   └── widgets.py                 # Streamlit UI components
└── data/
    └── twitter_metrics.csv        # Training data for impact model
```

## 🛠️ Installation & Setup

### 1. Clone and Navigate
```bash
git clone <repository-url>
cd newstrackr
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Impact Ranking Model
```bash
# Run the training notebook to generate the model
jupyter notebook notebooks/train_impact_ranker.ipynb
# Or run all cells to generate models/impact_ranker.pkl
```

### 4. Get NewsAPI Key (Optional)
- Sign up at [NewsAPI](https://newsapi.org/)
- Get your free API key
- The app works with sample data if no API key is provided

### 5. Run the Application
```bash
streamlit run app.py
```

## 🎯 Usage Guide

### Getting Started
1. **Launch the app**: Run `streamlit run app.py`
2. **Configure settings**: Use the sidebar to set your NewsAPI key (optional)
3. **Apply filters**: Select country, category, date range, and number of articles
4. **Browse news**: View ranked articles with impact scores
5. **Generate summaries**: Click "Generate Summary" for AI-powered article summaries

### Filter Options
- **📅 Date Range**: Select from and to dates
- **📊 Number of Articles**: 5-25 articles (slider)
- **🌍 Country**: 10 supported countries
- **🏷️ Category**: Business, Entertainment, Health, Science, Sports, Technology, General
- **⚙️ Advanced Filters**: Impact score threshold, engagement filters

### Features Walkthrough
- **Dashboard Overview**: Key metrics and charts
- **Article Cards**: Expandable cards with full details
- **AI Summaries**: Click to generate concise summaries
- **Social Metrics**: Simulated Twitter engagement data
- **Impact Scoring**: ML-based relevance ranking

## 🧠 ML Models Explained

### 1. News Summarizer
- **Model**: facebook/bart-large-cnn
- **Purpose**: Generate 2-paragraph summaries (~300 characters)
- **Features**: 
  - Handles long articles
  - Maintains context and key information
  - Includes source links
  - Cached for performance

### 2. Impact Ranker
- **Algorithm**: RandomForestRegressor
- **Features**:
  - TF-IDF vectorization of article titles (1000 features)
  - Twitter engagement metrics (likes, comments, retweets, views)
  - Recency scoring (time-based relevance)
  - Log-transformed metrics for better distribution
- **Training**: Synthetic data based on realistic engagement patterns
- **Output**: Impact score for article ranking

### Training Data Features
```python
Features used:
- TF-IDF vectors (1000 features from article titles)
- twitter_likes, twitter_comments, twitter_retweets, twitter_views
- log_likes, log_comments, log_retweets, log_views  
- hours_ago, recency_score
- Total: ~1010 features
```

## 🔧 Configuration

### Environment Variables (Optional)
Create a `.env` file:
```
NEWSAPI_KEY=your_newsapi_key_here
```

### Model Parameters
The impact ranking model uses these parameters:
- **n_estimators**: 100
- **max_depth**: 10
- **min_samples_split**: 5
- **min_samples_leaf**: 2

### Caching Settings
- **News data**: 1 hour cache
- **ML models**: Session-based loading
- **Summaries**: Cached per article content

## 📊 Data Sources

### News Data
- **Primary**: NewsAPI (with API key)
- **Fallback**: Sample articles (when no API key provided)
- **Coverage**: 10 countries, 7 categories

### Social Media Metrics
- **Source**: Simulated based on article characteristics
- **Factors**: Category, keywords, engagement patterns
- **Metrics**: Likes, comments, retweets, views, recency

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment
The app can be deployed on:
- **Streamlit Cloud**: Connect your GitHub repository
- **Heroku**: Use the Streamlit buildpack
- **AWS/GCP**: Deploy as a containerized application
- **Docker**: Create a Dockerfile for containerization

### Docker Deployment (Optional)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🔍 API Usage

### NewsAPI Integration
- **Endpoint**: Top headlines by country/category
- **Rate Limits**: 1000 requests/day (free tier)
- **Supported**: 10 countries, 7 categories

### Twitter API (Future Enhancement)
Currently uses simulated metrics. For real Twitter data:
- Integrate Twitter API v2
- Use free tier limits (300 requests/15 minutes)
- Fetch engagement metrics for news URLs

## 🧪 Testing

### Run Model Training
```bash
cd notebooks
jupyter notebook train_impact_ranker.ipynb
# Execute all cells to train and save the model
```

### Test Components
```python
# Test news fetching
from utils.fetch_live_news import get_news_fetcher
fetcher = get_news_fetcher()
articles = fetcher.fetch_news(country='us', category='technology')

# Test summarization
from utils.summarizer import get_summarizer
summarizer = get_summarizer()
summary = summarizer.summarize("Your article text here", "source_url")
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Make changes and test**
4. **Commit changes**: `git commit -m "Add new feature"`
5. **Push to branch**: `git push origin feature/new-feature`
6. **Create Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to functions
- Test new features thoroughly
- Update README for significant changes

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Common Issues
1. **Model not loading**: Ensure `models/impact_ranker.pkl` exists (run training notebook)
2. **API errors**: Check NewsAPI key validity and rate limits
3. **Memory issues**: Large models may require sufficient RAM

### Getting Help
- **Issues**: Create a GitHub issue
- **Questions**: Check the documentation
- **Contributions**: Follow the contributing guidelines

## 🔮 Future Enhancements

### Planned Features
- **Real Twitter API**: Replace simulated metrics
- **More ML Models**: Sentiment analysis, topic modeling
- **User Preferences**: Personalized news recommendations
- **Export Features**: PDF/Email summaries
- **Real-time Updates**: WebSocket-based live updates

### Technical Improvements
- **Database Integration**: PostgreSQL for article storage
- **API Endpoints**: REST API for external integrations
- **Better Caching**: Redis for distributed caching
- **Monitoring**: Application performance monitoring

---

**Built with ❤️ using Streamlit, Transformers, and Scikit-learn**