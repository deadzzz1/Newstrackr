"""
Streamlit Widgets for Newstrackr App
Contains sidebar filters and news display components
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pycountry
import plotly.graph_objects as go
import plotly.express as px

def create_sidebar_filters():
    """
    Create sidebar with filter options
    
    Returns:
        dict: Filter values selected by user
    """
    st.sidebar.title("🔍 Filters")
    
    # Date picker
    st.sidebar.subheader("📅 Date Range")
    date_from = st.sidebar.date_input(
        "From",
        value=datetime.now() - timedelta(days=1),
        max_value=datetime.now()
    )
    date_to = st.sidebar.date_input(
        "To",
        value=datetime.now(),
        max_value=datetime.now()
    )
    
    # Number of news slider
    st.sidebar.subheader("📊 Number of Articles")
    num_articles = st.sidebar.slider(
        "Select number of articles",
        min_value=5,
        max_value=25,
        value=10,
        step=1
    )
    
    # Country filter
    st.sidebar.subheader("🌍 Country")
    country_options = {
        'United States': 'us',
        'United Kingdom': 'gb',
        'Canada': 'ca',
        'Australia': 'au',
        'Germany': 'de',
        'France': 'fr',
        'Japan': 'jp',
        'India': 'in',
        'Brazil': 'br',
        'South Africa': 'za'
    }
    
    selected_country_name = st.sidebar.selectbox(
        "Select country",
        options=list(country_options.keys()),
        index=0
    )
    selected_country = country_options[selected_country_name]
    
    # Category selector
    st.sidebar.subheader("🏷️ Category")
    categories = {
        'General': 'general',
        'Business': 'business',
        'Entertainment': 'entertainment',
        'Health': 'health',
        'Science': 'science',
        'Sports': 'sports',
        'Technology': 'technology'
    }
    
    selected_category_name = st.sidebar.selectbox(
        "Select category",
        options=list(categories.keys()),
        index=0
    )
    selected_category = categories[selected_category_name]
    
    # Additional filters
    st.sidebar.subheader("⚙️ Advanced Filters")
    
    # Impact score threshold
    min_impact_score = st.sidebar.slider(
        "Minimum Impact Score",
        min_value=0,
        max_value=10000,
        value=0,
        step=100,
        help="Filter articles by minimum impact score"
    )
    
    # Twitter engagement threshold
    min_engagement = st.sidebar.slider(
        "Minimum Twitter Likes",
        min_value=0,
        max_value=5000,
        value=0,
        step=50,
        help="Filter articles by minimum Twitter likes"
    )
    
    return {
        'date_from': date_from,
        'date_to': date_to,
        'num_articles': num_articles,
        'country': selected_country,
        'country_name': selected_country_name,
        'category': selected_category,
        'category_name': selected_category_name,
        'min_impact_score': min_impact_score,
        'min_engagement': min_engagement
    }

def display_article_card(article, index, show_summary=False):
    """
    Display an individual article card
    
    Args:
        article (dict): Article data
        index (int): Article index
        show_summary (bool): Whether to show summary initially
    """
    # Create expandable container for each article
    with st.expander(f"📰 {article['title']}", expanded=show_summary):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Article details
            st.write(f"**Source:** {article['source']}")
            st.write(f"**Published:** {format_publish_time(article['publishedAt'])}")
            st.write(f"**Category:** {article['category'].title()}")
            
            # Description
            st.write("**Description:**")
            st.write(article['description'])
            
            # Twitter metrics
            st.write("**Social Media Engagement:**")
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
            
            # Article link
            st.markdown(f"**[Read Full Article]({article['url']})**")
        
        with col2:
            # Article image
            if article.get('urlToImage'):
                try:
                    st.image(article['urlToImage'], caption="Article Image", use_column_width=True)
                except:
                    st.info("Image not available")
            else:
                st.info("No image available")
            
            # Quick actions
            st.write("**Quick Actions:**")
            if st.button(f"📝 Summarize", key=f"summarize_{index}"):
                with st.spinner("Generating summary..."):
                    # This would trigger the summarization
                    st.session_state[f'show_summary_{index}'] = True
            
            if st.button(f"📊 Analytics", key=f"analytics_{index}"):
                # Show engagement analytics
                show_engagement_chart(article)

def show_engagement_chart(article):
    """
    Display engagement analytics chart for an article
    
    Args:
        article (dict): Article data
    """
    # Create engagement metrics chart
    metrics = ['Likes', 'Comments', 'Retweets', 'Views']
    values = [
        article['twitter_likes'],
        article['twitter_comments'], 
        article['twitter_retweets'],
        article['twitter_views']
    ]
    
    fig = go.Figure(data=[
        go.Bar(x=metrics, y=values, marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ])
    
    fig.update_layout(
        title="Social Media Engagement",
        xaxis_title="Metric",
        yaxis_title="Count",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_dashboard_metrics(articles):
    """
    Display dashboard overview metrics
    
    Args:
        articles (list): List of articles
    """
    if not articles:
        st.warning("No articles found with current filters")
        return
    
    # Calculate overview metrics
    total_articles = len(articles)
    avg_impact_score = sum(article['impact_score'] for article in articles) / total_articles
    total_engagement = sum(
        article['twitter_likes'] + article['twitter_comments'] + 
        article['twitter_retweets'] for article in articles
    )
    avg_engagement = total_engagement / total_articles
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📰 Total Articles",
            value=total_articles
        )
    
    with col2:
        st.metric(
            label="🎯 Avg Impact Score",
            value=f"{avg_impact_score:.0f}"
        )
    
    with col3:
        st.metric(
            label="📊 Total Engagement",
            value=f"{total_engagement:,}"
        )
    
    with col4:
        st.metric(
            label="💫 Avg Engagement",
            value=f"{avg_engagement:.0f}"
        )

def display_category_distribution(articles):
    """
    Display category distribution chart
    
    Args:
        articles (list): List of articles
    """
    if not articles:
        return
    
    # Count articles by category
    category_counts = {}
    for article in articles:
        category = article['category'].title()
        category_counts[category] = category_counts.get(category, 0) + 1
    
    # Create pie chart
    fig = px.pie(
        values=list(category_counts.values()),
        names=list(category_counts.keys()),
        title="Articles by Category"
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    st.plotly_chart(fig, use_container_width=True)

def display_impact_distribution(articles):
    """
    Display impact score distribution
    
    Args:
        articles (list): List of articles
    """
    if not articles:
        return
    
    impact_scores = [article['impact_score'] for article in articles]
    
    # Create histogram
    fig = px.histogram(
        x=impact_scores,
        nbins=20,
        title="Impact Score Distribution",
        labels={'x': 'Impact Score', 'y': 'Number of Articles'}
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def display_timeline_chart(articles):
    """
    Display timeline of articles
    
    Args:
        articles (list): List of articles
    """
    if not articles:
        return
    
    # Prepare data for timeline
    timeline_data = []
    for article in articles:
        timeline_data.append({
            'title': article['title'][:50] + "..." if len(article['title']) > 50 else article['title'],
            'publishedAt': pd.to_datetime(article['publishedAt']),
            'impact_score': article['impact_score'],
            'source': article['source']
        })
    
    df = pd.DataFrame(timeline_data)
    df = df.sort_values('publishedAt')
    
    # Create timeline chart
    fig = px.scatter(
        df, 
        x='publishedAt', 
        y='impact_score',
        hover_data=['title', 'source'],
        title="Article Timeline (Impact vs Time)",
        labels={'publishedAt': 'Published Time', 'impact_score': 'Impact Score'}
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def format_publish_time(timestamp_str):
    """
    Format publication timestamp for display
    
    Args:
        timestamp_str (str): ISO timestamp string
        
    Returns:
        str: Formatted time string
    """
    try:
        timestamp = pd.to_datetime(timestamp_str)
        now = pd.to_datetime(datetime.now())
        diff = now - timestamp
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return "Just now"
    except:
        return "Unknown time"

def display_search_bar():
    """
    Display search functionality
    
    Returns:
        str: Search query
    """
    search_query = st.text_input(
        "🔍 Search articles",
        placeholder="Enter keywords to search in titles and descriptions...",
        help="Search across article titles and descriptions"
    )
    
    return search_query

def filter_articles_by_search(articles, search_query):
    """
    Filter articles based on search query
    
    Args:
        articles (list): List of articles
        search_query (str): Search query
        
    Returns:
        list: Filtered articles
    """
    if not search_query:
        return articles
    
    search_query = search_query.lower()
    filtered_articles = []
    
    for article in articles:
        title_match = search_query in article['title'].lower()
        desc_match = search_query in article.get('description', '').lower()
        
        if title_match or desc_match:
            filtered_articles.append(article)
    
    return filtered_articles

def apply_filters(articles, filters):
    """
    Apply all filters to articles
    
    Args:
        articles (list): List of articles
        filters (dict): Filter parameters
        
    Returns:
        list: Filtered articles
    """
    filtered_articles = []
    
    for article in articles:
        # Check impact score filter
        if article['impact_score'] < filters['min_impact_score']:
            continue
            
        # Check engagement filter
        if article['twitter_likes'] < filters['min_engagement']:
            continue
            
        # Check date filter (simplified - would need proper date parsing)
        # For now, we'll include all articles
        
        filtered_articles.append(article)
    
    # Sort by impact score
    filtered_articles.sort(key=lambda x: x['impact_score'], reverse=True)
    
    # Limit to requested number
    return filtered_articles[:filters['num_articles']]