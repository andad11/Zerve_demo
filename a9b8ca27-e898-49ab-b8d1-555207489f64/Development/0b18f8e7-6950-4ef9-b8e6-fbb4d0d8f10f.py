import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import numpy as np

# Create histogram with KDE overlay for salary distribution
fig = go.Figure()

# Add histogram
fig.add_trace(go.Histogram(
    x=salaries_df['salary_in_usd'],
    name='Salary Distribution',
    nbinsx=50,
    marker_color='#636EFA',
    opacity=0.7,
    histnorm='probability density'
))

# Calculate KDE for smooth distribution overlay
salary_data = salaries_df['salary_in_usd'].dropna()
kde = stats.gaussian_kde(salary_data)
x_range = np.linspace(salary_data.min(), salary_data.max(), 200)
kde_values = kde(x_range)

# Add KDE curve
fig.add_trace(go.Scatter(
    x=x_range,
    y=kde_values,
    mode='lines',
    name='KDE',
    line=dict(color='red', width=2)
))

# Update layout
fig.update_layout(
    title='Salary Distribution with KDE Overlay',
    xaxis_title='Salary (USD)',
    yaxis_title='Density',
    hovermode='x unified',
    showlegend=True,
    template='plotly_white'
)

print(f"Created histogram with KDE for {len(salary_data)} salary records")
fig.show()