import plotly.express as px

# Get top 10 most common job titles for cleaner visualization
top_job_titles = salaries_df['job_title'].value_counts().head(10).index
filtered_df = salaries_df[salaries_df['job_title'].isin(top_job_titles)]

# Create violin plot by job title
fig_violin_job = px.violin(
    filtered_df,
    x='job_title',
    y='salary_in_usd',
    color='job_title',
    box=True,
    points='outliers',
    title='Salary Distribution by Job Title (Violin Plot - Top 10)',
    labels={
        'job_title': 'Job Title',
        'salary_in_usd': 'Salary (USD)'
    },
    template='plotly_white'
)

fig_violin_job.update_layout(
    showlegend=False,
    xaxis_tickangle=-45,
    hovermode='closest'
)

print(f"Created violin plot for top 10 job titles ({len(filtered_df)} records)")
fig_violin_job.show()