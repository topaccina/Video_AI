import dash
from dash import html
import dash_bootstrap_components as dbc

# Import custom components
from components.home_cards import card1, card2, card3

dash.register_page(__name__, path="/", name="Home", title="TBD | Home", order=0)


layout = dbc.Container(
    [
        # title
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            ["Video Discovery - Moving Faster with the AI "],
                            className="m-5 nav-container",
                        ),
                    ],
                    width=12,
                    className="row-titles",
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Container(
                    [
                        dbc.Col([card1], width="auto"),
                        dbc.Col([card2], width="auto"),
                        dbc.Col([card3], width="auto"),
                    ],
                    className="d-flex justify-content-center gap-5 nav-container",
                )
            ]
        ),
    ],
    className="align-center mt-5 ",
)
