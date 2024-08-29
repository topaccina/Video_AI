import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, name="FAQ", title="TBD | FAQ", order=3)


layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3(["Coming Soon"]),
                        html.P([html.B([""])], className="par"),
                    ],
                    width=12,
                    className="row-titles",
                )
            ]
        ),
    ]
)
