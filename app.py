from dash import Dash

app = Dash(__name__)

app.run_server(debug=True, use_reloader=False)
