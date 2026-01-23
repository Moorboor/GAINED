"""
GAINED - Therapy Session Analysis Application
Main entry point
"""
from dash import Dash, html, dcc, Input, Output

from src.ui_components import create_layout, create_sessions_layout
from src.callbacks import register_callbacks, register_clientside_callbacks, register_sessions_callbacks

# Initialize Dash app with external scripts for wavesurfer
external_scripts = [
    'https://unpkg.com/wavesurfer.js@7',
    'https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.min.js'
]

app = Dash(
    __name__,
    title='GAINED',
    suppress_callback_exceptions=True,
    external_scripts=external_scripts,
    assets_folder='assets'
)

# Multi-page layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Page routing callback
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/sessions':
        return create_sessions_layout()
    else:
        return create_layout()

# Register callbacks
register_callbacks(app)
register_clientside_callbacks(app)
register_sessions_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True, port=8050)
