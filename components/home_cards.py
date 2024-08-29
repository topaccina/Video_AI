import dash_bootstrap_components as dbc
from dash import html
import dash


# support functions to build components in the home page

learn = "https://icon-library.com/images/learn-icon-png/learn-icon-png-4.jpg"
compare = "https://cdn4.iconfinder.com/data/icons/business-businessman-employee-workers/261/employee-life-style-008-512.png"
search = (
    "https://cdn4.iconfinder.com/data/icons/men-holding/319/man-holding-017-1024.png"
)


def create_card(img, header, main, details=""):
    card = dbc.Card(
        [
            dbc.CardImg(
                src=img,
                top=True,
                style={
                    "width": "100%",
                    "height": "12vw",
                    "object-fit": "cover",
                    "align": "center",
                    "margin": "20px 10px 10px 10px",
                },
            ),
            dbc.CardBody(
                [
                    html.H3(header, className="text-primary"),
                    html.Div(main),
                    html.Div(details, className="small"),
                ]
            ),
            dbc.CardFooter(),
        ],
        className="shadow my-2 mt-5",
        style={"width": 360},
    )
    return card


card1 = create_card(search, "Search", "Get your Video by Topic ", details="")
card2 = create_card(
    compare,
    "Compare ",
    "Classify your Results",
    details="",
)
card3 = create_card(
    learn,
    "Learn ",
    "Keep Relevant Only ",
    details="",
)
