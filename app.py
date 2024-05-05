import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import numpy as np

import dash_bootstrap_components as dbc
from src.generate_data import generate_data
from src.gradient_descent import gradient_descent
from src.momentum_GD import momentum_GD
from src.nag import NAG
from src.adagrad import Adagrad
from src.rmsProp import RMSProp
from src.adam import Adam

data = generate_data(num_samples=100, num_features=2, noise=0.1)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SKETCHY], title='Optimization Algorithms Comparison')
server = app.server

app.layout = dbc.Container([
    html.Div([
        dbc.Row([
            dbc.Col([
                html.H1('Optimization Algorithms Comparison')
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.Label('Select Optimization Algorithm:'),
                dcc.Dropdown(
                    id='algorithm-dropdown',
                    options=[
                        {'label': 'Gradient Descent', 'value': 'Gradient Descent'},
                        {'label': 'Momentum Gradient Descent', 'value': 'Momentum'},
                        {'label': 'Nesterov Accelerated Gradient', 'value': 'NAG'},
                        {'label': 'Adagrad', 'value': 'Adagrad'},
                        {'label': 'RMSProp', 'value': 'RMSProp'},
                        {'label': 'Adam', 'value': 'Adam'}
                    ],
                    value='Gradient Descent'
                ),
                dbc.Row([
                    dbc.Col([
                        html.Div(id='features-container', children=[
                            html.Label('Number of Features'),
                            dbc.Input(
                                id='features',
                                type='number',
                                value=2,
                                step=1
                            )
                        ])
                    ], width=4),
                    dbc.Col([
                        html.Div(id='noise-container', children=[
                            html.Label('Noise'),
                            dbc.Input(
                                id='noise',
                                type='number',
                                value=0.1,
                                step=0.1
                            )
                        ])
                    ], width=4),
                    dbc.Col([
                        html.Div(id='num-samples-container', children=[
                            html.Label('Number of Samples'),
                            dbc.Input(
                                id='num-samples',
                                type='number',
                                value=100,
                                step=100
                            )
                        ])
                    ], width=4)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div(id='learning-rate-container', children=[
                            html.Label('Learning Rate'),
                            dbc.Input(
                                id='learning-rate',
                                type='number',
                                value=0.01,
                                step=0.01
                            )
                        ])
                    ], width=4),
                    dbc.Col([
                        html.Div(id='epochs-container', children=[
                            html.Label('Number of Epochs'),
                            dbc.Input(
                                id='epochs',
                                type='number',
                                value=100,
                                step=100
                            )
                        ])
                    ], width=4),
                    dbc.Col([
                        html.Div(id='batch-size-container', children=[
                            html.Label('Batch Size'),
                            dbc.Input(
                                id='batch-size',
                                type='number',
                                value=1,
                                step=16
                            )
                        ])
                    ], width=4)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div(id='momentum-container', children=[
                            html.Label('Momentum Coefficient'),
                            dbc.Input(
                                id='momentum',
                                type='number',
                                value=0.9,
                                step=0.1,
                                disabled=True
                            )
                        ])
                    ], width=6),
                    dbc.Col([
                        html.Div(id='epsilon-container', children=[
                            html.Label('Epsilon'),
                            dbc.Input(
                                id='epsilon',
                                type='number',
                                value=1e-8,
                                step=1e-8,
                                disabled=True
                            )
                        ])
                    ], width=6) 
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div(id='beta1-container', children=[
                            html.Label('Beta 1'),
                            dbc.Input(
                                id='beta1',
                                type='number',
                                value=0.1,
                                step=0.1,
                                disabled=True
                            )
                        ])
                    ]),
                    dbc.Col([
                        html.Div(id='beta2-container', children=[
                            html.Label('Beta 2'),
                            dbc.Input(
                                id='beta2',
                                type='number',
                                value=0.1,
                                step=0.1,
                                disabled=True
                            )
                        ])
                    ])
                ]),
                dbc.Button('Visualize', id='visualize-button', style={'margin-top': '10px'})
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='loss-graph')
            ]),
            dbc.Col([
                dcc.Graph(id='theta-graph')
            ])
        ])
    ])
])


@app.callback(
    [Output('loss-graph', 'figure'),
    Output('theta-graph', 'figure')],
    [Input('visualize-button', 'n_clicks')],
    [State('algorithm-dropdown', 'value'),
     State('learning-rate', 'value'),
     State('momentum', 'value'),
     State('batch-size', 'value'),
     State('epochs', 'value'),
     State('epsilon', 'value'),
     State('beta1', 'value'),
     State('beta2', 'value'),
     State('features', 'value'),
     State('noise', 'value'),
     State('num-samples', 'value')]
)
def update_graph(n_clicks, selected_algorithm, learning_rate, momentum, batch_size, epochs, epsilon, beta1, beta2, features, noise, num_samples):
    if not n_clicks:
        return px.line(), px.line()
    
    data = generate_data(num_samples=num_samples, num_features=features, noise=noise)

    if selected_algorithm == 'Gradient Descent':
        theta, thetas, loss, _, _ = gradient_descent(data[0], data[1], learning_rate=learning_rate, epochs=epochs, batch_size=batch_size)
    elif selected_algorithm == 'Momentum':
        theta, thetas, loss, _, _ = momentum_GD(data[0], data[1], learning_rate=learning_rate, momentum_term=momentum, epochs=epochs, batch_size=batch_size)
    elif selected_algorithm == 'NAG':
        theta, thetas, loss, _, _ = NAG(data[0], data[1], learning_rate=learning_rate, momentum_term=momentum, epochs=epochs, batch_size=batch_size)
    elif selected_algorithm == 'Adagrad':
        theta, thetas, loss, _, _ = Adagrad(data[0], data[1], learning_rate=learning_rate, epsilon=epsilon, epochs=epochs, batch_size=batch_size)
    elif selected_algorithm == 'RMSProp':
        theta, thetas, loss, _, _ = RMSProp(data[0], data[1], learning_rate=learning_rate, epsilon=epsilon, epochs=epochs, batch_size=batch_size , beta=beta1)
    elif selected_algorithm == 'Adam':
        theta, thetas, loss, _, _ = Adam(data[0], data[1], learning_rate=learning_rate, epsilon=epsilon, epochs=epochs, batch_size=batch_size, beta1=beta1, beta2=beta2)
    
    fig_loss = px.line(
        x=np.arange(len(loss)),
        y=loss,
        labels={'x': 'Iterations', 'y': 'Loss'}
    )
    
    fig_loss.update_layout(
        title='Loss vs Iterations',
        xaxis_title='Iterations',
        yaxis_title='Loss'
    )

    fig_loss.update_traces(mode='lines+markers')

    fig_theta = px.line()

    for i in range(features + 1):
        fig_theta.add_scatter(
            x=np.arange(len(thetas)),
            y=np.array(thetas)[:, i].reshape(-1),
            mode='lines',
            name=f'Theta {i}'
        )

    fig_theta.update_layout(
        title='Theta vs Iterations',
        xaxis_title='Iterations',
        yaxis_title='Theta'
    )

    return fig_loss, fig_theta

@app.callback(
    [Output('momentum', 'disabled'),
     Output('epsilon', 'disabled'),
     Output('beta1', 'disabled'),
     Output('beta2', 'disabled')],
    [Input('algorithm-dropdown', 'value')]
)
def update_hyperparameters(algorithm):

    if algorithm == 'Gradient Descent':
        return True, True, True, True
    elif algorithm == 'Momentum':
        return False, True, True, True
    elif algorithm == 'NAG':
        return False, True, True, True
    elif algorithm == 'Adagrad':
        return True, False, True, True
    elif algorithm == 'RMSProp':
        return True, False, False, True
    elif algorithm == 'Adam':
        return True, False, False, False

if __name__ == '__main__':
    app.run_server(debug=False)
