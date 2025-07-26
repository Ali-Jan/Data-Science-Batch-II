import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Global Superstore Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the dataset"""
    # Try common file paths
    file_paths = [
        'Global_Superstore.csv',
        '5. Data/Global_Superstore.csv',
        'data/Global_Superstore.csv'
    ]
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
    
    df = None
    used_path = None
    used_encoding = None
    
    for path in file_paths:
        for encoding in encodings:
            try:
                df = pd.read_csv(path, encoding=encoding)
                used_path = path
                used_encoding = encoding
                break
            except (FileNotFoundError, UnicodeDecodeError, UnicodeError):
                continue
        if df is not None:
            break
    
    if df is None:
        st.error("Could not load the CSV file. Please check if 'Global_Superstore.csv' exists and try uploading it again.")
        return None
    
    try:
        # Show success message with details
        # st.success(f"Data loaded from {used_path} using {used_encoding} encoding")
        
        # Convert Order Date
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        
        # Calculate Profit Margin if not present
        if 'Profit Margin' not in df.columns:
            df['Profit Margin'] = np.where(df['Sales'] != 0, 
                                         (df['Profit'] / df['Sales']) * 100, 
                                         0)
        
        # Remove any rows with invalid dates
        df = df.dropna(subset=['Order Date'])
        
        # Basic data validation
        required_columns = ['Region', 'Category', 'Sub-Category', 'Customer Name', 'Sales', 'Profit', 'Quantity']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.info("Available columns: " + ", ".join(df.columns.tolist()))
            return None
        
        return df
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.info("Please check your CSV file format and column names.")
        return None

def create_kpi_metrics(df):
    """Create KPI metrics"""
    total_sales = df['Sales'].sum()
    total_profit = df['Profit'].sum()
    total_orders = len(df)
    avg_profit_margin = df['Profit Margin'].mean()
    
    return total_sales, total_profit, total_orders, avg_profit_margin

def create_sales_trend_chart(df):
    """Create sales trend over time"""
    monthly_sales = df.groupby([df['Order Date'].dt.to_period('M')])['Sales'].sum().reset_index()
    monthly_sales['Order Date'] = monthly_sales['Order Date'].astype(str)
    
    fig = px.line(
        monthly_sales, 
        x='Order Date', 
        y='Sales',
        title='Sales Trend Over Time',
        labels={'Sales': 'Sales ($)', 'Order Date': 'Month'}
    )
    fig.update_layout(
        xaxis_tickangle=45,
        height=400
    )
    return fig

def create_region_performance_chart(df):
    """Create region-wise performance chart"""
    region_data = df.groupby('Region').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Sales by Region', 'Profit by Region'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    fig.add_trace(
        go.Bar(x=region_data['Region'], y=region_data['Sales'], name='Sales'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=region_data['Region'], y=region_data['Profit'], name='Profit'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_category_pie_chart(df):
    """Create category distribution pie chart"""
    category_sales = df.groupby('Category')['Sales'].sum().reset_index()
    
    fig = px.pie(
        category_sales,
        values='Sales',
        names='Category',
        title='Sales Distribution by Category'
    )
    fig.update_layout(height=400)
    return fig

def create_top_customers_chart(df):
    """Create top 5 customers by sales chart"""
    top_customers = df.groupby('Customer Name')['Sales'].sum().nlargest(5).reset_index()
    
    fig = px.bar(
        top_customers,
        x='Sales',
        y='Customer Name',
        orientation='h',
        title='Top 5 Customers by Sales',
        labels={'Sales': 'Total Sales ($)', 'Customer Name': 'Customer'}
    )
    fig.update_layout(height=400)
    return fig

def create_profitability_scatter(df):
    """Create sales vs profit scatter plot"""
    fig = px.scatter(
        df,
        x='Sales',
        y='Profit',
        color='Category',
        size='Quantity',
        title='Sales vs Profit Analysis',
        labels={'Sales': 'Sales ($)', 'Profit': 'Profit ($)'}
    )
    fig.update_layout(height=400)
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">Global Superstore Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_and_prepare_data()
    
    if df is None:
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Region filter
    regions = ['All'] + sorted(df['Region'].unique().tolist())
    selected_region = st.sidebar.selectbox('Select Region:', regions)
    
    # Category filter
    categories = ['All'] + sorted(df['Category'].unique().tolist())
    selected_category = st.sidebar.selectbox('Select Category:', categories)
    
    # Sub-category filter
    if selected_category != 'All':
        sub_categories = ['All'] + sorted(df[df['Category'] == selected_category]['Sub-Category'].unique().tolist())
    else:
        sub_categories = ['All'] + sorted(df['Sub-Category'].unique().tolist())
    selected_sub_category = st.sidebar.selectbox('Select Sub-Category:', sub_categories)
    
    # Date range filter
    min_date = df['Order Date'].min().date()
    max_date = df['Order Date'].max().date()
    date_range = st.sidebar.date_input(
        'Select Date Range:',
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]
    
    if selected_sub_category != 'All':
        filtered_df = filtered_df[filtered_df['Sub-Category'] == selected_sub_category]
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['Order Date'].dt.date >= start_date) & 
            (filtered_df['Order Date'].dt.date <= end_date)
        ]
    
    # Display filtered data info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Records shown:** {len(filtered_df):,}")
    st.sidebar.markdown(f"**Total records:** {len(df):,}")
    
    # Main dashboard content
    if len(filtered_df) == 0:
        st.warning("No data available for the selected filters.")
        return
    
    # KPI Section
    st.markdown("## Key Performance Indicators")
    
    total_sales, total_profit, total_orders, avg_profit_margin = create_kpi_metrics(filtered_df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Sales",
            value=f"${total_sales:,.2f}",
            delta=f"{len(filtered_df)} orders"
        )
    
    with col2:
        st.metric(
            label="Total Profit",
            value=f"${total_profit:,.2f}",
            delta=f"{total_profit/total_sales*100:.1f}% margin"
        )
    
    with col3:
        st.metric(
            label="Total Orders",
            value=f"{total_orders:,}",
            delta=f"${total_sales/total_orders:.2f} avg"
        )
    
    with col4:
        st.metric(
            label="Avg Profit Margin",
            value=f"{avg_profit_margin:.1f}%",
            delta="Per order"
        )
    
    # Charts Section
    st.markdown("---")
    st.markdown("## Performance Analysis")
    
    # Row 1: Sales trend and region performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_sales_trend_chart(filtered_df), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_region_performance_chart(filtered_df), use_container_width=True)
    
    # Row 2: Category distribution and top customers
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_category_pie_chart(filtered_df), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_top_customers_chart(filtered_df), use_container_width=True)
    
    # Row 3: Profitability analysis
    st.plotly_chart(create_profitability_scatter(filtered_df), use_container_width=True)
    
    # Data Table Section
    st.markdown("---")
    st.markdown("## Detailed Data")
    
    # Show summary statistics
    with st.expander("Summary Statistics"):
        st.write(filtered_df.describe())
    
    # Show top 10 records
    with st.expander("Sample Data (Top 10 Records)"):
        display_columns = ['Order Date', 'Region', 'Category', 'Sub-Category', 
                          'Customer Name', 'Sales', 'Profit', 'Profit Margin']
        st.dataframe(
            filtered_df[display_columns].head(10),
            use_container_width=True
        )
    
    # Download option
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name=f"superstore_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
