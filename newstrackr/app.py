"""
Newstrackr - News Aggregator and Summarizer
Main Streamlit Application
"""

import streamlit as st
import os
import sys
from datetime import datetime
import pandas as pd

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.fetch_live_news import get_news_fetcher
from utils.summarizer import get_summarizer, summarize_cached
from utils.widgets import (
    create_sidebar_filters, display_article_card, display_dashboard_metrics,
    display_category_distribution, display_impact_distribution, 
    display_timeline_chart, display_search_bar, filter_articles_by_search,
    apply_filters
)

# Page configuration
st.set_page_config(
    page_title="Newstrackr - News Aggregator & Summarizer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .article-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: white;
    }
    
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'news_data' not in st.session_state:
        st.session_state.news_data = []
    if 'last_fetch_time' not in st.session_state:
        st.session_state.last_fetch_time = None
    if 'current_filters' not in st.session_state:
        st.session_state.current_filters = {}

def load_news_data(filters, api_key=None):
    """Load news data with caching"""
    # Check if we need to refresh data (cache for 1 hour)
    current_time = datetime.now()
    
    if (st.session_state.last_fetch_time is None or 
        (current_time - st.session_state.last_fetch_time).seconds > 3600 or
        st.session_state.current_filters != filters):
        
        with st.spinner("🔄 Fetching latest news..."):
            # Get news fetcher
            news_fetcher = get_news_fetcher(api_key)
            
            # Fetch news
            articles = news_fetcher.fetch_news(
                country=filters['country'],
                category=filters['category'],
                page_size=25  # Fetch more than needed for filtering
            )
            
            st.session_state.news_data = articles
            st.session_state.last_fetch_time = current_time
            st.session_state.current_filters = filters.copy()
    
    return st.session_state.news_data

def display_article_with_summary(article, index):
    """Display article with optional summary"""
    with st.expander(f"📰 {article['title']}", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Article details
            st.write(f"**Source:** {article['source']}")
            st.write(f"**Published:** {pd.to_datetime(article['publishedAt']).strftime('%Y-%m-%d %H:%M')}")
            st.write(f"**Category:** {article['category'].title()}")
            
            # Description
            st.write("**Description:**")
            st.write(article['description'])
            
            # Social media metrics
            st.write("**📊 Social Media Engagement:**")
            col_likes, col_comments, col_retweets, col_views = st.columns(4)
            
            with col_likes:
                st.metric("👍 Likes", f"{article['twitter_likes']:,}")
            with col_comments:
                st.metric("💬 Comments", f"{article['twitter_comments']:,}")
            with col_retweets:
                st.metric("🔄 Retweets", f"{article['twitter_retweets']:,}")
            with col_views:
                st.metric("👁️ Views", f"{article['twitter_views']:,}")
            
            # Impact score
            st.metric("🎯 Impact Score", f"{article['impact_score']:.0f}")
            
            # Summary section
            if st.button(f"📝 Generate Summary", key=f"summarize_{index}"):
                with st.spinner("Generating AI summary..."):
                    try:
                        # Use description as content for summarization
                        content = f"{article['title']}. {article['description']}"
                        summary = summarize_cached(content, article['url'])
                        
                        st.markdown("### 📄 AI Summary")
                        st.markdown(summary)
                        
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
            
            # Article link
            st.markdown(f"**[📖 Read Full Article]({article['url']})**")
        
        with col2:
            # Article image
            if article.get('urlToImage'):
                try:
                    st.image(article['urlToImage'], caption="Article Image", use_column_width=True)
                except:
                    st.info("📷 Image not available")
            else:
                st.info("📷 No image available")
            
            # Engagement analytics
            st.write("**📈 Quick Stats:**")
            total_engagement = (article['twitter_likes'] + 
                              article['twitter_comments'] + 
                              article['twitter_retweets'])
            
            st.metric("Total Engagement", f"{total_engagement:,}")
            st.metric("Engagement Rate", f"{(total_engagement/max(article['twitter_views'], 1)*100):.1f}%")

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📰 Newstrackr</h1>
        <p>AI-Powered News Aggregator and Summarizer</p>
        <p>Discover, rank, and summarize news with ML-driven impact scoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar filters
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input(
        "NewsAPI Key (Optional)",
        type="password",
        help="Enter your NewsAPI key for live data. Leave empty to use sample data."
    )
    
    # Get filters
    filters = create_sidebar_filters()
    
    # Add refresh button
    if st.sidebar.button("🔄 Refresh News", type="primary"):
        st.session_state.last_fetch_time = None
        st.experimental_rerun()
    
    # Display last update time
    if st.session_state.last_fetch_time:
        st.sidebar.info(f"Last updated: {st.session_state.last_fetch_time.strftime('%H:%M:%S')}")
    
    # Load news data
    try:
        raw_articles = load_news_data(filters, api_key)
        
        # Apply filters
        filtered_articles = apply_filters(raw_articles, filters)
        
        # Search functionality
        search_query = display_search_bar()
        if search_query:
            filtered_articles = filter_articles_by_search(filtered_articles, search_query)
        
        # Main content area
        if not filtered_articles:
            st.warning("No articles found matching your criteria. Try adjusting the filters.")
            return
        
        # Dashboard metrics
        st.markdown("## 📊 Dashboard Overview")
        display_dashboard_metrics(filtered_articles)
        
        # Charts section
        col1, col2 = st.columns(2)
        
        with col1:
            display_category_distribution(filtered_articles)
        
        with col2:
            display_impact_distribution(filtered_articles)
        
        # Timeline chart
        display_timeline_chart(filtered_articles)
        
        # Articles section
        st.markdown("## 📰 Top News Articles")
        st.markdown(f"*Showing {len(filtered_articles)} articles ranked by impact score*")
        
        # Display articles
        for i, article in enumerate(filtered_articles):
            display_article_with_summary(article, i)
        
        # Footer information
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🤖 AI Models Used")
            st.markdown("""
            - **Summarizer**: facebook/bart-large-cnn
            - **Impact Ranker**: TF-IDF + RandomForest
            """)
        
        with col2:
            st.markdown("### 📊 Data Sources")
            st.markdown("""
            - **News**: NewsAPI
            - **Engagement**: Simulated Twitter metrics
            - **Ranking**: ML-based impact scoring
            """)
        
        with col3:
            st.markdown("### 🔄 Caching")
            st.markdown("""
            - **News data**: 1 hour cache
            - **Summaries**: Cached per article
            - **Model**: Loaded once per session
            """)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.markdown("Please check your API key and try again, or contact support if the issue persists.")

if __name__ == "__main__":
    main()