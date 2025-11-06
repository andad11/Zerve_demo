import plotly.express as px

# Create box plot by experience level
fig_exp = px.box(
    salaries_df,
    x='experience_level',
    y='salary_in_usd',
    color='experience_level',
    title='Salary Distribution by Experience Level',
    labels={
        'experience_level': 'Experience Level',
        'salary_in_usd': 'Salary (USD)'
    },
    category_orders={
        'experience_level': ['EN', 'MI', 'SE', 'EX']
    },
    template='plotly_white'
)

fig_exp.update_layout(
    showlegend=False,
    hovermode='closest'
)

print(f"Created box plot by experience level")
fig_exp.show()