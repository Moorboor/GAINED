"""
GAINED - Therapy Session Analysis Application
Main entry point
"""
from dash import Dash

from src.ui_components import create_layout
from src.callbacks import register_callbacks, register_clientside_callbacks

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

# Set layout
app.layout = create_layout()

# Register callbacks
register_callbacks(app)
register_clientside_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True, port=8050)
