#!/usr/bin/env python3
"""
Standalone script to train the impact ranking model
Run this to generate the models/impact_ranker.pkl file
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def generate_training_data():
    """Generate synthetic training data"""
    print("Generating synthetic training data...")
    
    np.random.seed(42)
    
    # Sample news titles
    news_titles = [
        "Breaking: Major earthquake hits California",
        "Stock market reaches all-time high",
        "New COVID variant discovered",
        "Celebrity wedding announcement",
        "Tech company announces layoffs",
        "Political scandal emerges",
        "Sports team wins championship",
        "New scientific discovery published",
        "Climate change report released",
        "Economic recession warning issued",
        "Space mission successful launch",
        "Healthcare breakthrough announced",
        "Local news: School fundraiser event",
        "International trade agreement signed",
        "Cryptocurrency market crash",
        "Social media platform update",
        "Weather warning issued",
        "Entertainment industry news",
        "Education policy changes",
        "Environmental protection measures"
    ] * 50  # Repeat to have more data
    
    n_samples = len(news_titles)
    
    # Generate synthetic Twitter metrics
    twitter_likes = np.random.exponential(1000, n_samples)
    twitter_comments = np.random.exponential(100, n_samples)
    twitter_retweets = np.random.exponential(200, n_samples)
    twitter_views = np.random.exponential(10000, n_samples)
    
    # Generate recency (hours ago)
    hours_ago = np.random.exponential(12, n_samples)
    
    # Create impact score based on engagement and recency
    impact_score = (
        twitter_likes * 0.3 + 
        twitter_comments * 0.4 + 
        twitter_retweets * 0.5 + 
        twitter_views * 0.1 +
        np.maximum(0, 24 - hours_ago) * 100  # Recency bonus
    ) + np.random.normal(0, 500, n_samples)  # Add noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'title': news_titles,
        'twitter_likes': twitter_likes,
        'twitter_comments': twitter_comments,
        'twitter_retweets': twitter_retweets,
        'twitter_views': twitter_views,
        'hours_ago': hours_ago,
        'impact_score': impact_score
    })
    
    return df

def train_model():
    """Train the impact ranking model"""
    print("Training impact ranking model...")
    
    # Generate training data
    df = generate_training_data()
    
    # Feature Engineering
    print("Creating TF-IDF features from titles...")
    
    # TF-IDF vectorization of titles
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        lowercase=True
    )
    
    title_tfidf = tfidf_vectorizer.fit_transform(df['title'])
    title_features = pd.DataFrame(
        title_tfidf.toarray(), 
        columns=[f'tfidf_{i}' for i in range(title_tfidf.shape[1])]
    )
    
    # Combine with Twitter metrics and recency
    twitter_features = df[['twitter_likes', 'twitter_comments', 'twitter_retweets', 'twitter_views', 'hours_ago']].copy()
    
    # Log transform Twitter metrics to reduce skewness
    twitter_features['log_likes'] = np.log1p(twitter_features['twitter_likes'])
    twitter_features['log_comments'] = np.log1p(twitter_features['twitter_comments'])
    twitter_features['log_retweets'] = np.log1p(twitter_features['twitter_retweets'])
    twitter_features['log_views'] = np.log1p(twitter_features['twitter_views'])
    
    # Create recency score (more recent = higher score)
    twitter_features['recency_score'] = np.maximum(0, 24 - twitter_features['hours_ago'])
    
    # Combine all features
    all_features = pd.concat([title_features, twitter_features], axis=1)
    target = df['impact_score']
    
    print(f"Feature matrix shape: {all_features.shape}")
    
    # Split data and train model
    X_train, X_test, y_train, y_test = train_test_split(
        all_features, target, test_size=0.2, random_state=42
    )
    
    # Scale features for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest Regressor
    print("Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_train = rf_model.predict(X_train_scaled)
    y_pred_test = rf_model.predict(X_test_scaled)
    
    # Evaluate model
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\nModel Performance:")
    print(f"Train MSE: {train_mse:.2f}")
    print(f"Test MSE: {test_mse:.2f}")
    print(f"Train R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    
    # Save the trained model and preprocessors
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Create a pipeline dictionary to save all components
    model_pipeline = {
        'tfidf_vectorizer': tfidf_vectorizer,
        'scaler': scaler,
        'model': rf_model,
        'feature_names': all_features.columns.tolist()
    }
    
    # Save the complete pipeline
    joblib.dump(model_pipeline, 'models/impact_ranker.pkl')
    print("Model pipeline saved to models/impact_ranker.pkl")
    
    # Save training data for future reference
    df.to_csv('data/twitter_metrics.csv', index=False)
    print("Training data saved to data/twitter_metrics.csv")
    
    # Test the saved model
    print("\nTesting the saved model...")
    
    # Load the model
    loaded_pipeline = joblib.load('models/impact_ranker.pkl')
    
    # Test with a sample news article
    test_title = "Breaking: Major tech company announces revolutionary AI breakthrough"
    test_metrics = {
        'twitter_likes': 1500,
        'twitter_comments': 200,
        'twitter_retweets': 800,
        'twitter_views': 15000,
        'hours_ago': 2
    }
    
    # Create test dataframe
    test_df = pd.DataFrame({
        'title': [test_title],
        'twitter_likes': [test_metrics['twitter_likes']],
        'twitter_comments': [test_metrics['twitter_comments']],
        'twitter_retweets': [test_metrics['twitter_retweets']],
        'twitter_views': [test_metrics['twitter_views']],
        'hours_ago': [test_metrics['hours_ago']]
    })
    
    # Transform features using saved pipeline
    test_tfidf = loaded_pipeline['tfidf_vectorizer'].transform(test_df['title'])
    test_title_features = pd.DataFrame(
        test_tfidf.toarray(), 
        columns=[f'tfidf_{i}' for i in range(test_tfidf.shape[1])]
    )
    
    test_twitter_features = test_df[['twitter_likes', 'twitter_comments', 'twitter_retweets', 'twitter_views', 'hours_ago']].copy()
    test_twitter_features['log_likes'] = np.log1p(test_twitter_features['twitter_likes'])
    test_twitter_features['log_comments'] = np.log1p(test_twitter_features['twitter_comments'])
    test_twitter_features['log_retweets'] = np.log1p(test_twitter_features['twitter_retweets'])
    test_twitter_features['log_views'] = np.log1p(test_twitter_features['twitter_views'])
    test_twitter_features['recency_score'] = np.maximum(0, 24 - test_twitter_features['hours_ago'])
    
    test_features = pd.concat([test_title_features, test_twitter_features], axis=1)
    test_features_scaled = loaded_pipeline['scaler'].transform(test_features)
    
    # Predict impact score
    predicted_impact = loaded_pipeline['model'].predict(test_features_scaled)[0]
    
    print(f"\nTest Article: {test_title}")
    print(f"Twitter Metrics: {test_metrics}")
    print(f"Predicted Impact Score: {predicted_impact:.2f}")
    print("\nModel training and testing completed successfully!")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("Please make sure you have the required dependencies installed:")
        print("pip install pandas numpy scikit-learn joblib")