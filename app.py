'''
This is the main script for the dashboard. It will be used to create the layout of the dashboard.
This will call pages in the container view.
'''
from dash import Dash, html, page_container
import dash_bootstrap_components as dbc
import dash

# Initialize the app
app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           use_pages=True)

server = app.server

# Navigation bar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Dashboard", href="/", style={'font-weight': 'bold'})),
    ],
    brand_href="/",
    sticky="top",
    color="#c2aeec",
    dark=True,)
    
# LAYOUT PAGE   
app.layout = html.Div([
    dbc.Container([
        html.H1(children='Stock Prices Analysis with Average Price Line', style={'font-weight': 'bold'}),
        html.H5(children='Are there an Optimal Investment Strategy?'),
        html.H5(children='Author: Yuting Weng'),
        html.Br(),
        # Navigation bar
        html.Div(navbar),
        html.Br(),
        dash.page_container
    ])
])  

# Run
if __name__ == '__main__':
    app.run(debug=True, port=8052)
