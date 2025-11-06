import plotly.express as px

# Create violin plot by experience level
fig_violin_exp = px.violin(
    salaries_df,
    x='experience_level',
    y='salary_in_usd',
    color='experience_level',
    box=True,
    points='outliers',
    title='Salary Distribution by Experience Level (Violin Plot)',
    labels={
        'experience_level': 'Experience Level',
        'salary_in_usd': 'Salary (USD)'
    },
    category_orders={
        'experience_level': ['EN', 'MI', 'SE', 'EX']
    },
    template='plotly_white'
)

fig_violin_exp.update_layout(
    showlegend=False,
    hovermode='closest'
)

print("Created violin plot by experience level")
fig_violin_exp.show()