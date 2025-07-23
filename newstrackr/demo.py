#!/usr/bin/env python3
"""
Newstrackr Demo Script
Demonstrates the key features and capabilities of the news aggregator
"""

import sys
import os
sys.path.append('utils')

def demo_news_fetching():
    """Demonstrate news fetching capabilities"""
    print("📰 Demo: News Fetching")
    print("-" * 40)
    
    try:
        from fetch_live_news import get_news_fetcher
        
        # Create news fetcher (without API key for demo)
        fetcher = get_news_fetcher()
        
        # Fetch sample news
        articles = fetcher.fetch_news(country='us', category='technology', page_size=5)
        
        print(f"✅ Successfully fetched {len(articles)} articles")
        print("\n📊 Sample articles with impact scores:")
        
        for i, article in enumerate(articles[:3], 1):
            print(f"\n{i}. {article['title'][:60]}...")
            print(f"   📈 Impact Score: {article['impact_score']:.0f}")
            print(f"   👍 Likes: {article['twitter_likes']:,}")
            print(f"   🔄 Retweets: {article['twitter_retweets']:,}")
            print(f"   📊 Views: {article['twitter_views']:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in news fetching demo: {str(e)}")
        return False

def demo_summarization():
    """Demonstrate summarization capabilities"""
    print("\n📝 Demo: AI Summarization")
    print("-" * 40)
    
    try:
        from summarizer import get_summarizer
        
        # Sample article text
        sample_text = """
        Major technology companies are investing heavily in artificial intelligence 
        research and development. Recent breakthroughs in machine learning have led 
        to significant advances in natural language processing, computer vision, and 
        autonomous systems. These developments are expected to transform multiple 
        industries including healthcare, finance, and transportation. Experts predict 
        that AI will continue to evolve rapidly, with new applications emerging across 
        various sectors. However, concerns about ethical AI development and potential 
        job displacement remain significant topics of discussion among policymakers 
        and industry leaders.
        """
        
        # Get summarizer (note: this might take time to load the model)
        print("🤖 Loading BART model for summarization...")
        summarizer = get_summarizer()
        
        # Generate summary
        print("🔄 Generating summary...")
        summary = summarizer.summarize(
            sample_text, 
            "https://example.com/ai-article"
        )
        
        print("✅ Summary generated successfully!")
        print("\n📄 Original text:")
        print(sample_text.strip())
        print(f"\n📝 AI Summary ({len(summary)} characters):")
        print(summary)
        
        return True
        
    except Exception as e:
        print(f"❌ Error in summarization demo: {str(e)}")
        print("Note: BART model requires significant resources and may not work in all environments")
        return False

def demo_impact_scoring():
    """Demonstrate impact scoring model"""
    print("\n🎯 Demo: Impact Scoring Model")
    print("-" * 40)
    
    try:
        # Check if model file exists
        if not os.path.exists('models/impact_ranker.pkl'):
            print("⚠️  Impact ranking model not found. Training it now...")
            from train_model import train_model
            train_model()
        
        import joblib
        import pandas as pd
        import numpy as np
        
        # Load the trained model
        model_pipeline = joblib.load('models/impact_ranker.pkl')
        print("✅ Impact ranking model loaded successfully!")
        
        # Test with different article scenarios
        test_articles = [
            {
                'title': "Breaking: Major earthquake hits California",
                'twitter_likes': 5000,
                'twitter_comments': 800,
                'twitter_retweets': 2000,
                'twitter_views': 50000,
                'hours_ago': 1
            },
            {
                'title': "Celebrity shares breakfast photo",
                'twitter_likes': 1200,
                'twitter_comments': 150,
                'twitter_retweets': 300,
                'twitter_views': 8000,
                'hours_ago': 3
            },
            {
                'title': "Scientific breakthrough in cancer research",
                'twitter_likes': 2500,
                'twitter_comments': 400,
                'twitter_retweets': 1200,
                'twitter_views': 25000,
                'hours_ago': 2
            }
        ]
        
        print("\n📊 Impact scores for different articles:")
        
        for i, article in enumerate(test_articles, 1):
            # Create test dataframe
            test_df = pd.DataFrame([article])
            
            # Transform features
            test_tfidf = model_pipeline['tfidf_vectorizer'].transform(test_df['title'])
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
            test_features_scaled = model_pipeline['scaler'].transform(test_features)
            
            # Predict impact score
            impact_score = model_pipeline['model'].predict(test_features_scaled)[0]
            
            print(f"\n{i}. {article['title']}")
            print(f"   🎯 Impact Score: {impact_score:.0f}")
            print(f"   📊 Engagement: {article['twitter_likes']+article['twitter_comments']+article['twitter_retweets']:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in impact scoring demo: {str(e)}")
        return False

def demo_app_features():
    """Demonstrate app features"""
    print("\n🎨 Demo: Streamlit App Features")
    print("-" * 40)
    
    print("✅ Key Features Available:")
    print("📰 Real-time news fetching with NewsAPI integration")
    print("🤖 AI-powered summarization using BART transformer")
    print("🎯 ML-based impact scoring with TF-IDF + RandomForest")
    print("🔍 Advanced filtering (date, country, category, engagement)")
    print("📊 Interactive dashboards and visualizations")
    print("💾 Smart caching (1-hour for news, session for models)")
    print("🌍 Multi-country support (10 countries)")
    print("🏷️  Multi-category support (7 categories)")
    
    print("\n🎯 To run the full Streamlit app:")
    print("   streamlit run app.py")
    
    print("\n📱 App will include:")
    print("   • Modern UI with custom CSS styling")
    print("   • Expandable news cards with detailed metrics")
    print("   • One-click AI summarization")
    print("   • Real-time search and filtering")
    print("   • Social media engagement analytics")
    print("   • Timeline and distribution charts")

def main():
    """Run the complete demo"""
    print("🚀 Newstrackr - News Aggregator & Summarizer Demo")
    print("=" * 60)
    print("This demo showcases the key features of the Newstrackr application")
    print("=" * 60)
    
    # Track demo results
    results = []
    
    # Run demos
    results.append(demo_news_fetching())
    results.append(demo_impact_scoring())
    
    # Skip summarization demo in limited environments
    print("\n⚠️  Skipping AI summarization demo due to model size constraints")
    print("   (BART model requires ~1.6GB and significant computational resources)")
    
    demo_app_features()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Demo Summary")
    print("=" * 60)
    
    successful_demos = sum(results)
    total_demos = len(results)
    
    print(f"✅ {successful_demos}/{total_demos} core features demonstrated successfully")
    
    if successful_demos == total_demos:
        print("🎉 All core features are working correctly!")
        print("\n🚀 Ready to run: streamlit run app.py")
    else:
        print("⚠️  Some features may need additional setup")
        print("   Run: python3 setup.py for automatic setup")
    
    print("\n📚 For detailed instructions, see README.md")
    print("🔗 For issues or contributions, visit the GitHub repository")

if __name__ == "__main__":
    main()