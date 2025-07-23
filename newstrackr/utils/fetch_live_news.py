"""
Live News Fetcher with Twitter Metrics
Fetches real-time news using NewsAPI and simulates Twitter metrics
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from newsapi import NewsApiClient
import random
import time
import joblib
import os

class LiveNewsFetcher:
    def __init__(self, api_key=None):
        """
        Initialize news fetcher
        
        Args:
            api_key (str): NewsAPI key
        """
        self.api_key = api_key
        self.newsapi = None
        if api_key:
            self.newsapi = NewsApiClient(api_key=api_key)
        
        # Load impact ranker model if available
        self.impact_model = None
        self._load_impact_model()
    
    def _load_impact_model(self):
        """Load the trained impact ranking model"""
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'impact_ranker.pkl')
        try:
            if os.path.exists(model_path):
                self.impact_model = joblib.load(model_path)
                print("Impact ranking model loaded successfully")
            else:
                print("Impact ranking model not found - using fallback scoring")
        except Exception as e:
            print(f"Error loading impact model: {e}")
            self.impact_model = None
    
    def simulate_twitter_metrics(self, title, category="general"):
        """
        Simulate Twitter engagement metrics based on article characteristics
        
        Args:
            title (str): Article title
            category (str): News category
            
        Returns:
            dict: Simulated Twitter metrics
        """
        # Base metrics influenced by category
        category_multipliers = {
            'business': {'likes': 1.2, 'comments': 1.0, 'retweets': 1.1, 'views': 1.3},
            'entertainment': {'likes': 1.8, 'comments': 1.5, 'retweets': 1.6, 'views': 2.0},
            'general': {'likes': 1.0, 'comments': 1.0, 'retweets': 1.0, 'views': 1.0},
            'health': {'likes': 1.3, 'comments': 1.4, 'retweets': 1.2, 'views': 1.1},
            'science': {'likes': 0.9, 'comments': 1.1, 'retweets': 0.8, 'views': 0.9},
            'sports': {'likes': 1.6, 'comments': 1.3, 'retweets': 1.4, 'views': 1.7},
            'technology': {'likes': 1.1, 'comments': 1.2, 'retweets': 1.0, 'views': 1.2}
        }
        
        multiplier = category_multipliers.get(category, category_multipliers['general'])
        
        # Keywords that drive engagement
        high_engagement_keywords = [
            'breaking', 'exclusive', 'shocking', 'revealed', 'urgent', 'crisis',
            'scandal', 'breakthrough', 'historic', 'unprecedented', 'warning',
            'emergency', 'viral', 'trending', 'investigation', 'exposed'
        ]
        
        title_lower = title.lower()
        engagement_boost = 1.0
        for keyword in high_engagement_keywords:
            if keyword in title_lower:
                engagement_boost += 0.3
        
        # Base metrics with some randomness
        base_likes = random.randint(50, 2000)
        base_comments = random.randint(10, 400)
        base_retweets = random.randint(20, 800)
        base_views = random.randint(500, 20000)
        
        # Apply multipliers and engagement boost
        metrics = {
            'twitter_likes': int(base_likes * multiplier['likes'] * engagement_boost),
            'twitter_comments': int(base_comments * multiplier['comments'] * engagement_boost),
            'twitter_retweets': int(base_retweets * multiplier['retweets'] * engagement_boost),
            'twitter_views': int(base_views * multiplier['views'] * engagement_boost),
            'hours_ago': random.uniform(0.5, 12.0)  # How many hours ago it was posted
        }
        
        return metrics
    
    def calculate_impact_score(self, title, twitter_metrics):
        """
        Calculate impact score using the trained model or fallback method
        
        Args:
            title (str): Article title
            twitter_metrics (dict): Twitter engagement metrics
            
        Returns:
            float: Impact score
        """
        if self.impact_model:
            try:
                # Prepare features for the model
                test_df = pd.DataFrame({
                    'title': [title],
                    'twitter_likes': [twitter_metrics['twitter_likes']],
                    'twitter_comments': [twitter_metrics['twitter_comments']],
                    'twitter_retweets': [twitter_metrics['twitter_retweets']],
                    'twitter_views': [twitter_metrics['twitter_views']],
                    'hours_ago': [twitter_metrics['hours_ago']]
                })
                
                # Transform features
                tfidf = self.impact_model['tfidf_vectorizer'].transform(test_df['title'])
                title_features = pd.DataFrame(
                    tfidf.toarray(), 
                    columns=[f'tfidf_{i}' for i in range(tfidf.shape[1])]
                )
                
                twitter_features = test_df[['twitter_likes', 'twitter_comments', 'twitter_retweets', 'twitter_views', 'hours_ago']].copy()
                twitter_features['log_likes'] = np.log1p(twitter_features['twitter_likes'])
                twitter_features['log_comments'] = np.log1p(twitter_features['twitter_comments'])
                twitter_features['log_retweets'] = np.log1p(twitter_features['twitter_retweets'])
                twitter_features['log_views'] = np.log1p(twitter_features['twitter_views'])
                twitter_features['recency_score'] = np.maximum(0, 24 - twitter_features['hours_ago'])
                
                all_features = pd.concat([title_features, twitter_features], axis=1)
                features_scaled = self.impact_model['scaler'].transform(all_features)
                
                # Predict impact score
                impact_score = self.impact_model['model'].predict(features_scaled)[0]
                return float(impact_score)
                
            except Exception as e:
                print(f"Error using trained model: {e}")
                # Fall back to simple calculation
        
        # Fallback impact calculation
        likes = twitter_metrics['twitter_likes']
        comments = twitter_metrics['twitter_comments']
        retweets = twitter_metrics['twitter_retweets']
        views = twitter_metrics['twitter_views']
        hours_ago = twitter_metrics['hours_ago']
        
        # Simple weighted score with recency bonus
        impact_score = (
            likes * 0.3 + 
            comments * 0.4 + 
            retweets * 0.5 + 
            views * 0.1 +
            max(0, 24 - hours_ago) * 100
        )
        
        return impact_score
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def fetch_news(_self, country='us', category='general', page_size=20):
        """
        Fetch news from NewsAPI with caching
        
        Args:
            country (str): Country code (e.g., 'us', 'gb')
            category (str): News category
            page_size (int): Number of articles to fetch
            
        Returns:
            list: List of news articles with metadata
        """
        if not _self.newsapi:
            # Return sample data if no API key
            return _self._get_sample_news(page_size)
        
        try:
            # Fetch top headlines
            top_headlines = _self.newsapi.get_top_headlines(
                country=country,
                category=category,
                page_size=page_size
            )
            
            articles = []
            for article in top_headlines['articles']:
                # Skip articles without title or description
                if not article.get('title') or not article.get('description'):
                    continue
                
                # Simulate Twitter metrics
                twitter_metrics = _self.simulate_twitter_metrics(
                    article['title'], 
                    category
                )
                
                # Calculate impact score
                impact_score = _self.calculate_impact_score(
                    article['title'], 
                    twitter_metrics
                )
                
                # Prepare article data
                article_data = {
                    'title': article['title'],
                    'description': article['description'],
                    'url': article['url'],
                    'urlToImage': article.get('urlToImage'),
                    'publishedAt': article['publishedAt'],
                    'source': article['source']['name'],
                    'category': category,
                    'country': country,
                    'impact_score': impact_score,
                    **twitter_metrics
                }
                
                articles.append(article_data)
            
            # Sort by impact score
            articles.sort(key=lambda x: x['impact_score'], reverse=True)
            
            return articles
            
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")
            return _self._get_sample_news(page_size)
    
    def _get_sample_news(self, page_size=20):
        """Return sample news data when API is not available"""
        sample_articles = [
            {
                'title': 'Breaking: Major tech company announces AI breakthrough',
                'description': 'A leading technology company has announced a significant advancement in artificial intelligence that could revolutionize the industry.',
                'url': 'https://example.com/ai-breakthrough',
                'urlToImage': 'https://via.placeholder.com/400x200',
                'publishedAt': datetime.now().isoformat(),
                'source': 'Tech News',
                'category': 'technology',
                'country': 'us'
            },
            {
                'title': 'Stock market reaches new high amid economic optimism',
                'description': 'Financial markets continue their upward trajectory as investors show confidence in economic recovery.',
                'url': 'https://example.com/market-high',
                'urlToImage': 'https://via.placeholder.com/400x200',
                'publishedAt': (datetime.now() - timedelta(hours=2)).isoformat(),
                'source': 'Financial Times',
                'category': 'business',
                'country': 'us'
            },
            {
                'title': 'Climate summit reaches historic agreement',
                'description': 'World leaders agree on ambitious new targets for carbon emissions reduction at international climate conference.',
                'url': 'https://example.com/climate-agreement',
                'urlToImage': 'https://via.placeholder.com/400x200',
                'publishedAt': (datetime.now() - timedelta(hours=4)).isoformat(),
                'source': 'Environmental News',
                'category': 'general',
                'country': 'us'
            }
        ] * (page_size // 3 + 1)
        
        # Add simulated metrics to sample articles
        for i, article in enumerate(sample_articles[:page_size]):
            twitter_metrics = self.simulate_twitter_metrics(
                article['title'], 
                article['category']
            )
            impact_score = self.calculate_impact_score(
                article['title'], 
                twitter_metrics
            )
            
            article.update(twitter_metrics)
            article['impact_score'] = impact_score
        
        # Sort by impact score
        sample_articles.sort(key=lambda x: x['impact_score'], reverse=True)
        
        return sample_articles[:page_size]
    
    def get_full_article_content(self, url):
        """
        Fetch full article content from URL (simplified version)
        In production, you might want to use a more sophisticated scraper
        
        Args:
            url (str): Article URL
            
        Returns:
            str: Article content
        """
        try:
            response = requests.get(url, timeout=10)
            # This is a simplified version - in production you'd parse HTML properly
            # For now, return the description as content
            return "Full article content would be extracted from the URL. This is a placeholder for the actual web scraping implementation."
        except:
            return "Unable to fetch full article content."

# Global instance
news_fetcher_instance = None

def get_news_fetcher(api_key=None):
    """Get or create news fetcher instance"""
    global news_fetcher_instance
    if news_fetcher_instance is None:
        news_fetcher_instance = LiveNewsFetcher(api_key)
    return news_fetcher_instance