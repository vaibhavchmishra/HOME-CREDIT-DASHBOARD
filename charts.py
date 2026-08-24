import plotly.express as px
def histogram(df,col,color=None): return px.histogram(df,x=col,color=color,nbins=40,title=f'Distribution of {col}')
def scatter(df,x,y,color=None): return px.scatter(df,x=x,y=y,color=color,opacity=.5,title=f'{y} vs {x}')
