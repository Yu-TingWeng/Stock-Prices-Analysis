'''
This file contains the dashboard page.
'''
import dash
from dash import html, dcc, callback, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(r"C:\Users\user\Documents\GitHub\Stock-Prices-Analysis")
import analyze_stock_prices as stock

dash.register_page(__name__, path='/')

# import Data
df_sp500 = stock.scrape_stock('%5EGSPC', 
                        '2000-01-01', 
                        '2023-12-08'
                        )
df_dow = stock.scrape_stock('%5EDJI?p=%5EDJI', 
                      '2000-01-01', 
                      '2023-12-08'
                      )
df_nasdaq = stock.scrape_stock('^IXIC', 
                         '2000-01-01', 
                         '2023-12-08'
                         )


controls = dbc.Card(
    [
        html.Div(
            [
                dbc.Label("Choose a Stock"),
                dcc.Dropdown(
                    id="stock_name",
                    options=[
                        {"label": name, "value": name} for name in ['S&P 500', 'Dow Jones', 'NASDAQ']
                    ],
                    value='S&P 500',
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("From What Year to Now"),
                dcc.Dropdown(
                    id="start_year",
                    options=[
                        {"label": str(year), "value": str(year)} for year in range(2000, 2024)
                    ],
                    value= "2000",
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("N-Day Average Price"),
                dcc.Dropdown(
                    id="n_day",
                    options=[
                        {"label": str(day), "value": str(day)} for day in ["10", "20", "60"]
                    ],
                    value="10",
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Choose a Strategy"),
                dcc.Dropdown(
                    id="strategy",
                    options=[
                        {"label": number, "value": number} for number in ["Strategy 1", "Strategy 2", "Strategy 3"]
                    ],
                    value="Strategy 1",
                ),
            ]
        ),
    ],
    body=True,
)



# LAYOUT PAGE
layout = html.Div([
    html.H5(children="The purpose of this analysis is to...", style={'color': '#c2aeec', 'line-height': '2'}),
    html.Div([
        # Dropdowns Menu
        dbc.Row([
            dbc.Col(controls, md=4, style={'margin-top': '20px'}),
            dbc.Col(dcc.Graph(id='line_plot')),
        ]),

        html.H5(children='BUY & SELL Date matched with Strategy'),
        html.Div(dash_table.DataTable(id='data-table', 
                                      page_size=10, 
                                      page_action='native', 
                                      style_cell={'textAlign': 'left'}, 
                                      style_data={'whiteSpace': 'normal', 'height': 'auto'}, 
                                      style_table={'overflowX': 'auto'})),
        
        html.H5(children='Mean Profit & Return Rate of the Strategy'),
        html.Div(dash_table.DataTable(id='stat-table', 
                                      style_cell={'textAlign': 'left'},
                                      style_data={'whiteSpace': 'normal', 'height': 'auto'},
                                      style_table={'overflowX': 'auto'})),
        html.Br(),
        html.Br(),
    ]),
])


# CALLBACKS
@callback(
    [Output('line_plot', 'figure'),
     Output('data-table', 'data'),
     Output('stat-table', 'data')],
    [Input('stock_name', 'value'),
     Input('start_year', 'value'),
     Input('n_day', 'value'),
     Input('strategy', 'value')]
)   
def update_output(selected_stock, selected_year, selected_n_day, selected_strategy):
    # Filter data based on selected options
    if selected_stock == 'S&P 500':
        df = df_sp500
    elif selected_stock == 'Dow Jones':
        df = df_dow
    else:
        df= df_nasdaq

    # Assuming 'Date' is one of your columns, you can filter by year
    df_filtered = df[df['Date'].dt.year >= int(selected_year)]
    
    # Call the average_calculate function with the selected_n_day value
    df_filtered = stock.average_calculate(df_filtered, int(selected_n_day))
    
    # Initialize with default values
    strategy_profit = pd.DataFrame()
    strategy_stat = pd.DataFrame()
    
    #STRATEGY
    if selected_strategy == 'Strategy 1':
        strategy_profit, strategy_stat = stock.strategy1_profit_statistics(df_filtered, 'now_avg_compare')  
    elif selected_strategy == 'Strategy 2':
        strategy_profit, strategy_stat = stock.strategy2_profit_statistics(df_filtered, 'Average Signal', 'now_avg_compare')
    elif selected_strategy == 'Strategy 3':
        strategy_profit, strategy_stat = stock.strategy3_profit_statistics(df_filtered, 'Average Signal', 'now_avg_compare')

    # Create DataTable components for data and statistics
    data_table = strategy_profit.to_dict('records')
    stat_table = strategy_stat.to_dict('records')
    
    fig = px.line(df_filtered, x='Date', y='Close', title=f'{selected_stock} Stock Price')
    fig.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['Average'], mode='lines', name='Average Price'))
    
    return fig, data_table, stat_table


# https://dash-bootstrap-components.opensource.faculty.ai/examples/iris/


