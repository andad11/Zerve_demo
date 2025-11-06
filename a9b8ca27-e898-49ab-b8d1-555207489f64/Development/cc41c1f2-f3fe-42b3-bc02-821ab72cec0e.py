import plotly.express as px

# Get top 10 most common job titles for cleaner visualization
top_jobs = salaries_df['job_title'].value_counts().head(10).index
filtered_salary_df = salaries_df[salaries_df['job_title'].isin(top_jobs)]

# Create box plot by job title
fig_job = px.box(
    filtered_salary_df,
    x='job_title',
    y='salary_in_usd',
    color='job_title',
    title='Salary Distribution by Job Title (Top 10)',
    labels={
        'job_title': 'Job Title',
        'salary_in_usd': 'Salary (USD)'
    },
    template='plotly_white'
)

fig_job.update_layout(
    showlegend=False,
    xaxis_tickangle=-45,
    hovermode='closest'
)

print(f"Created box plot for top 10 job titles ({len(filtered_salary_df)} records)")
fig_job.show()