import dash_bootstrap_components as dbc
from dash import html, dcc
import dash


# support functions to build components in the search  page -- video details cards
def create_videoCard_default(img, header, main="", details=""):
    card = dbc.Card(
        [
            dbc.CardImg(
                src=img,
                top=True,
                style={
                    "width": "50%",
                    "height": "6vw",
                    "object-fit": "cover",
                    "align": "center",
                    "margin": "auto ",
                },
                className="mt-2",
            ),
            dbc.CardBody(
                [
                    html.H4(header, className="text-primary text-center"),
                    html.Div(main),
                    html.Div(details, className="small"),
                ]
            ),
            dbc.CardFooter(),
        ],
        className="shadow my-2 mt-5",
    )
    return card


def create_videoCard(img, header, data, details=""):
    description = data.videoDescription.loc[0]
    author = data.author.loc[0]
    timeStamp = data.videoTimeStamp.loc[0]
    duration = data.duration.loc[0]
    views = data.view.loc[0]
    videoLink = data.videoLink.loc[0]
    summary = data.summary.loc[0]
    card = dbc.Card(
        [
            # dbc.CardHeader(),
            dbc.CardImg(
                src=img,
                top=True,
                style={
                    "width": "30%",
                    # "height": "12vw",
                    "object-fit": "cover",
                    "align": "center",
                    "margin": "auto ",
                },
                className="mt-2",
            ),
            dbc.CardBody(
                [
                    html.H4(header, className="text-primary text-center"),
                    dbc.Label(["Description:"], className="fw-bold"),
                    html.P(description),
                    dbc.Label(["Author:"], className="fw-bold"),
                    html.P(author),
                    dbc.Label(["Published at:"], className="fw-bold"),
                    html.P(timeStamp),
                    dbc.Label(["Duration:"], className="fw-bold"),
                    html.P(f"{duration} sec"),
                    dbc.Label("Views count"),
                    html.P(views),
                    dbc.Label("Video url"),
                    html.P(
                        dcc.Link(
                            href=videoLink,
                            target="_blank",
                        )
                    ),
                    dbc.Label("AI generated Summary"),
                    html.P(summary),
                ],
                className="shadow  m-5 p-3",
            ),
            dbc.CardFooter(),
        ],
        className="shadow  m-2",
        # style={"width": 360},
    )
    return card
