# Import packages
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

import plotly.graph_objs as go
import plotly.express as px

# Import files
crypto_owners = pd.read_csv('data\crypto_owners.csv')
crypto_price = pd.read_csv('data\DF_pp.csv')
crypto_mktcap = pd.read_csv('data\MKTCAP.csv')

total = float(crypto_mktcap['Market Cap_'].sum())

topcontinent = crypto_owners.drop(columns=['country', 'pop_crypto','perc_pop','iso3']).\
    groupby(['continent']).sum().sort_values(['crypto_owners'], ascending=False).reset_index()

# Create Visualizations
y_value_options = [ {'label':'Volume ', 'value':'Volume'},
                    {'label': 'USD_price', 'value': 'price_usd'},
                    {'label': 'Candles', 'value': 'Candle'}       ]

crypto_options = [
    dict(label='Crypto ' + crypto, value=crypto)
    for crypto in crypto_price['Name'].unique()]

continent_options = [
    dict(label=country, value=country)
    for country in crypto_owners['continent'].unique()]

year_slider = dcc.RangeSlider(
        id='year_slider',
        min=2010,
        max=2022,
        value=[2021, 2022],
        marks={ '2010': 'Year 2010',
                '2011': 'Year 2011',
                '2012': 'Year 2012',
                '2013': 'Year 2013',
                '2014': 'Year 2014',
                '2015': 'Year 2015',
                '2016': 'Year 2016',
                '2017': 'Year 2017',
                '2018': 'Year 2018',
                '2019': 'Year 2019',
                '2020': 'Year 2020',
                '2021': 'Year 2021',
                '2022': 'Year 2022'},
        step=1
    )


fact1='The highest Total Cryptocurrency Market Cap was 2.97 Trillion US Dollars'
fact2='Excluding Bitcoin the highest Total Cryptocurrency Market Cap was 1.64 Trillion US Dollars'
fact3='The lowest Bitcoin Dominance of the Total Cryptocurrency Market Cap was 36%'

facts=[]
facts.append(fact1)
facts.append(fact2)
facts.append(fact3)

tree_map = px.treemap(crypto_mktcap, path=['Crypto Name'],values='Market Cap_', hover_data =['%'])


# Create the app
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    #1
    html.Div([
        html.H1(children='CRYPTO DASHBOARD'),
        html.Img(src=app.get_asset_url('novaims.png'),
                 style={'position': 'relative', 'width': '4%', 'left': '-10px', 'top': '-75px'}),
    ], className='top_bar'),

    #2
    html.Div([
        #2a
        html.Div([

            #2a1
            html.Div([

                #2a1a
                html.Div([

                    #2a1a1
                    html.Div([
                        html.Label('Choose a cryptocurrency:'),
                        dcc.Dropdown(
                            id='crypto_drop',
                            options=crypto_options,
                            value='bitcoin'),

                        html.Label('Choose an attribute:'),
                        dcc.Dropdown(
                            id='y_value_drop',
                            options=y_value_options,
                            value='Candle')
                    ], className='box', style={'display': 'inline-block', 'vertical-align': 'center',
                                               'margin-left': '0.5vw', 'margin-top': '3vw','width':'25%'}),

                    #2a1a2
                    html.Div([
                        html.Div([
                                # 2a1a2a
                                html.Div([
                                    html.H4('Max Price', style={"text-align": "center", 'font-weight': 'normal'}),
                                    html.H3(id='max_price_value', style={"text-align": "center"})
                                ], className='box_crypto'),

                                # 2a1a2b
                                html.Div([
                                    html.H4("Min Price", style={"text-align": "center", "font-weight": "normal"}),
                                    html.H3(id="min_price_value", style={"text-align": "center"})
                                ], className='box_crypto'),

                                # 2a1a2c
                                html.Div([
                                    html.H4("ICO date", style={"text-align": "center", "font-weight": "normal"}),
                                    html.H3(id="min_date", style={"text-align": "center"})
                                ], className='box_crypto'),

                                # 2a1a2d
                                html.Div([
                                    html.H4("Best day variation", style={"text-align": "center", "font-weight": "normal"}),
                                    html.H3(id="best1", style={"text-align": "center"})
                                ], className='box_crypto')
                        ], style={'display':'flex'})
                    ], className='box', style={'display': 'inline-block', 'vertical-align': 'top',
                                               'margin-left': '2vw', 'margin-top': '3vw', 'width':'50%'})

                ]),

                #2a1b
                html.Div([
                    html.Label('Price and Volume chart', style={"font-weight": "bold",'font-size':'large'}),
                    dcc.Graph(id='grafico', style={'border-radius': '20px'}),
                    year_slider
                ], className='box')
            ], style={'display': 'inline-block','width':'50%','margin-top': '3vw'}),

            #2a2
            html.Div([

                #2a2a
                html.Div([
                    html.Div([
                        html.H4(id='fact_index',style={"text-align": "center",
                                                       "font-weight": "bold", 'font-size': 15}),
                        html.Br(),
                        html.Button('Click', id='button', n_clicks=3)
                    ], className='box_fact')
                ], className='box'),
                #2a2b
                html.Div([
                    html.Label('Crypto Top 20 Market Cap',
                               style={"font-weight": "bold",'font-size':'large'}),
                    html.Br(),
                    html.Label('Total Market Cap: '+ str(round((total /1000000000) , 3)) + ' T USD',
                               style={ 'font-size': 'small'}),

                    dcc.Graph(id='tree_map', figure= tree_map),
                    html.Br(),
                    html.Div([
                        html.H3('This data was obtained on the 5th of March')
                    ], className='box_fact')
                ], className='box', style={'margin-top': '1.5vw'})
            ], style={'display': 'inline-block','width':'50%','vertical-align': '-2700%'})
        ]),

        #2b
        html.Div([

            #2b1
            html.Div([
                dcc.Dropdown(
                    id='continents',
                    options=continent_options,
                    multi=True,
                    value=['Europe','South America','North America','Asia','Africa','Oceania','North America'],
                    placeholder='Select a continent',
                    searchable=False)
            ]),

            #2b2
            html.Div([

                #2b2a
                html.Div([
                    html.Br(),
                    dbc.Table.from_dataframe(topcontinent, bordered=True, className='box_fact')
                ]),

                #2b2b
                html.Div([
                    html.Br(),
                    dcc.Graph(id='mapa_crypto1', style={'position':'relative'})
                ], style={'width':'80%'})
            ], style={'display':'flex'})
        ], className='box', style={'margin-top': '1.5vw','margin-left': '0.5vw'})
    ], className='main')
])

@app.callback(
    Output('fact_index', 'children'),
    Input(component_id='button', component_property='n_clicks')

)

def update_fact(n_clicks):
    return facts[int(n_clicks % 3)]


@app.callback(
    [Output('grafico', 'figure'),
     Output('max_price_value', 'children'),
     Output('min_price_value', 'children'),
     Output('min_date', 'children'),
     Output('best1', 'children')],
    [Input('crypto_drop', 'value'),
     Input('year_slider', 'value'),
     Input('y_value_drop', 'value')])

# Update charts
def update_graph(cryptos,year,y_value):

    dff = crypto_price[crypto_price['Name'] == cryptos]
    dfff = dff[(dff['year'] >= year[0]) & (dff['year'] <= year[1])]

    max_price = round(float(dfff.groupby(['Name']).max()['price_usd']), 2)
    min_price = round(float(dfff.groupby(['Name']).min()['price_usd']), 2)
    min_date = (dff.groupby(['Name']).min()['Date'])
    best_1d= str (round(float(dfff[dfff['Name']== cryptos].groupby('Name').max()['pct_ch_1'])*100,2)) +' %'


    if y_value== 'Candle' :
        fig = go.Figure(data=[go.Candlestick(x=dfff['Date'],
                                             open=dfff['Open'],
                                             high=dfff['High'],
                                             low=dfff['Low'],
                                             close=dfff['Close'])])
    else:
        for crypto in cryptos:
            fig = px.line(dfff, x="Date", y=y_value, color='Name', height=600)

        fig.update_layout(yaxis={'title':y_value},
                          paper_bgcolor='#f9f9f9',
                          plot_bgcolor='white')


    return fig, max_price, min_price, min_date, best_1d

@app.callback(
    Output('mapa_crypto1','figure'),
    [Input('continents','value')])

def update_map(x):
    df_owners = crypto_owners[crypto_owners['continent'].isin(list(x))]

    crypto_map = dict(type='choropleth',
                       locations = df_owners['country'],
                       locationmode='country names',
                       autocolorscale = True,
                       z=df_owners['perc_pop'],
                       colorscale = 'inferno',
                       marker_line_color= 'rgba(0,0,0,0)',
                       colorbar= {'title':'% Crypto users'},
                       colorbar_lenmode='fraction',
                       colorbar_len=0.8,
                       colorbar_x=1,
                       colorbar_xanchor='left',
                       colorbar_y=0.5,
                       name='')



    layout_choropleth = dict(geo=dict(scope='world',
                                      projection={'type':'natural earth'},
                                      bgcolor='rgba(0,0,0,0)',
                                      showframe=False
                                      ),
                             title=dict(
                                 text='<b>Percentage of the population that uses crypto</b>',
                                 x=0.5
                             ),
                             margin=dict(l=0,
                                         r=0,
                                         b=0,
                                         t=30,
                                         pad=0),
                             paper_bgcolor='rgba(0,0,0,0)',
                             plot_bgcolor='rgba(0,0,0,0)'
                             )

    show_map = go.Figure(data=crypto_map,layout=layout_choropleth)

    show_map.update_layout(margin=dict(l=60, r=60, t=50, b=50))

    return show_map


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)