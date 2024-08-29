from dash import Dash, html
import dash
import dash_bootstrap_components as dbc
from dotenv import load_dotenv
import os


from langchain_community.document_loaders import YoutubeLoader


# get env var
load_dotenv()
##########################################################
YOUTUBE_DATA_API_KEY = os.getenv("YOUTUBE_DATA_API_KEY")
api_version = "v3"
api_service_name = "youtube"
###########################################################
app = Dash(
    __name__,
    use_pages=True,
    pages_folder="pages",
    external_stylesheets=[dbc.themes.MORPH, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    # prevent_initial_callbacks=True,
    # prevent_initial_callbacks="initial_duplicate",
)
side_nav = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Container(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [html.H1(["TBD"], className="fw-bold danger")],
                                ),
                            ],
                            align="center",
                        )
                    ],
                    className=" logo mt-3 mb-3",
                )
            ],
        ),
        dbc.Row(
            [
                dbc.Nav(
                    [
                        dbc.NavLink(page["name"], active="exact", href=page["path"])
                        for page in dash.page_registry.values()
                    ],
                    vertical=True,
                    pills=True,
                    className="my-nav",
                ),
            ],
            className="m-3",
        ),
    ],
    className=" d-flex flex-column align-content-start m-3 nav-container vh-100",
)


############################################################################################


############################################################################################
# App Layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col([side_nav], width=2),
                dbc.Col([dbc.Row([dash.page_container])], width=10),
            ]
        ),
        dbc.Row(
            [
                dbc.Col([], width=2),
                dbc.Col([dbc.Row([html.Hr()])], width=10),
            ]
        ),
    ],
    fluid=True,
)

############################################################################################
# Run App
if __name__ == "__main__":
    app.run_server(debug=True)
