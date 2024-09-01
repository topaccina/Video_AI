import dash
from dash import html, dcc, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import dash_ag_grid as dag


# YouTube Data API
import googleapiclient.discovery

# for loading env variables
from dotenv import load_dotenv
import os


import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from datetime import date


# langchain packages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings

#
# for data dimensionality reduction
from sklearn.decomposition import PCA

# custom functions
from components.video_card import create_videoCard, create_videoCard_default
from utils.support import getEnvVar
from utils.support import get_comments, get_video_info, count_tokens


# import env var to run youTube, Openai api
load_dotenv()
YOUTUBE_DATA_API_KEY = os.getenv("YOUTUBE_DATA_API_KEY")
api_version = "v3"
api_service_name = "youtube"
OPENAI_API_KEY = getEnvVar()

# init youTube API service
youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=YOUTUBE_DATA_API_KEY
)


dash.register_page(__name__, name="Search", title="TBD | Search", order=2)

# placeholder dataframes - for initial app setup before running any search
df = pd.DataFrame(
    data={
        "videoTitle": ["No results. Setup and run a search"],
        "author": [""],
        "videoTimeStamp": [""],
        "duration": [""],
        "views": [""],
    }
)

df_default = df.copy()
#########################################################################
# define components to build the page
button_group = dbc.Container(
    [
        dbc.RadioItems(
            id="radios",
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {"label": "short", "value": 1},
                {"label": "medium", "value": 2},
                # {"label": "Option 3", "value": 3},
            ],
            value=1,
        ),
        # html.Div(id="output"),
    ],
    className="radio-group",
)
# dropdown_menu_items = [
#     dbc.DropdownMenuItem("NOT ACTIVE MENU", id="dropdown-menu-item-notificaton"),
#     dbc.DropdownMenuItem("short - less 4 mins", id="dropdown-menu-item-short"),
#     dbc.DropdownMenuItem("medium - up to 10 mins", id="dropdown-menu-item-medium"),
# ]

# define columns to be shown in ag-grid table
columnDefs = [
    {
        "headerName": "video Index",
        "field": "index",
        "checkboxSelection": True,
    },
    {
        "headerName": "video Title",
        "field": "videoTitle",
    },
    {"headerName": "author", "field": "author"},
    {"headerName": "videoTimeStamp", "field": "videoTimeStamp"},
    {"headerName": "duration (sec)", "field": "duration"},
    {"headerName": "views", "field": "view"},
]

# control panel definition. it includes all the controls to define the query.
control_panel = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Container(
                            [
                                dbc.Label(" Video Topic"),
                                dbc.Input(
                                    placeholder="Keyword here...",
                                    type="text",
                                    id="input-topic",
                                ),
                            ]
                        )
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dbc.Container(
                            [
                                dbc.Label("Published After"),
                                dbc.Input(
                                    type="date",
                                    step=1,
                                    value=date.today(),
                                    id="input-date",
                                ),
                            ],
                        )
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dbc.Container(
                            [
                                dbc.Label("Max Results Count"),
                                dbc.Input(
                                    type="number",
                                    min=1,
                                    max=10,
                                    step=1,
                                    value=10,
                                    style={"width": "100px"},
                                    id="input-count",
                                ),
                                # dbc.FormText("Max 10 "),
                            ],
                        )
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dbc.Container(
                            [
                                dbc.Label("Duration"),
                                # dbc.DropdownMenu(dropdown_menu_items, label="Options"),
                                button_group,
                            ],
                            id="input-duration",
                            className="p-0",
                        )
                    ],
                    width=3,
                ),
            ]
        ),
    ],
    className="mt-3  mb-3 d-flex justify-content-around control ",
)
# main layout page
layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        # this is to store search results -data persists also w/ refresh page
                        dcc.Store(id="memory-output", storage_type="session"),
                    ]
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3(["Search Settings"], className="mt-5 mb-0 "),
                    ],
                    width=12,
                    className="row-titles",
                )
            ]
        ),
        dbc.Row([dbc.Col([control_panel])]),
        dbc.Row(
            [
                dbc.Col([dbc.Button("Search", id="btn-search")], width=2),
                dbc.Col([dbc.Button("Analyze", id="btn-analysis")]),
            ]
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Loading(
                            [
                                dbc.Container(
                                    [
                                        dag.AgGrid(
                                            id="crossfilter-example",
                                            rowData=df.reset_index().to_dict("records"),
                                            columnDefs=columnDefs,
                                            defaultColDef={
                                                "resizable": True,
                                                "sortable": True,
                                                "filter": True,
                                            },
                                            columnSize="sizeToFit",
                                            dashGridOptions={
                                                "rowSelection": "single",
                                                "overflow": "hidden",
                                            },
                                            rowStyle={
                                                "backgroundColor": "#f0f5fa",
                                                "color": "#7b8ab8",
                                            },
                                        )
                                    ],
                                    className="d-flex justify-content-around ",
                                    id="ag-grid-container",
                                ),
                            ],
                            type="circle",
                        ),
                    ]
                )
            ]
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Tabs(
                            [
                                dbc.Tab(
                                    dbc.Container(
                                        [
                                            create_videoCard_default(
                                                "https://static.vecteezy.com/system/resources/previews/023/986/473/original/youtube-logo-youtube-logo-transparent-youtube-icon-transparent-free-free-png.png",
                                                "select one video from the table",
                                            )
                                        ],
                                        id="id-video-card",
                                    ),
                                    label="Details",
                                ),
                                dbc.Tab(
                                    dbc.Container([], id="id-cluster"), label="Compare"
                                ),
                                dbc.Tab(
                                    dbc.Container([], id="id-sentiment"),
                                    label="Sentiment",
                                ),
                            ]
                        ),
                    ]
                )
            ]
        ),
    ],
    className="mt-5",
)


# callback to retrieve last search data stored in dcc.store in case we return on search page OR refresh the search page
@callback(
    Output("ag-grid-container", "children"),
    Input("memory-output", "data"),
)
def fillTable(data):
    dfff = df_default
    if data is None:
        dfff = df_default
    else:
        dfff = pd.DataFrame(data)
    # repopulate the table
    agTable = dag.AgGrid(
        id="crossfilter-example",
        rowData=dfff.reset_index().to_dict("records"),
        columnDefs=columnDefs,
        defaultColDef={
            "resizable": True,
            "sortable": True,
            "filter": True,
        },
        columnSize="sizeToFit",
        dashGridOptions={
            "rowSelection": "single",
            "overflow": "hidden",
        },
        rowStyle={
            "backgroundColor": "#f0f5fa",
            "color": "#7b8ab8",
        },
    )
    return agTable


# run semantic cluster and sentiment analysis - this is to re-run analysis on existing data after the page refresh.
@callback(
    Output(
        "id-cluster",
        "children",
    ),
    Output(
        "id-sentiment",
        "children",
    ),
    Input("btn-analysis", "n_clicks"),
    State("memory-output", "data"),
    prevent_initial_call=True,
)
def analyze(n, data):
    dff = df_default
    if data is None:
        dff = df_default
    else:
        dff = pd.DataFrame(data)

    if "transcript_embed_pca_x" in dff.columns.tolist():
        fig = px.scatter(
            dff.reset_index(),
            x="transcript_embed_pca_x",
            y="transcript_embed_pca_y",
            hover_data=["videoTitle"],
            size="view_resize",
            color="sentiment_avg",
            color_continuous_scale=px.colors.sequential.RdBu_r,
        )

        fig.update_layout(
            showlegend=True,
            margin=dict(t=10, l=10, b=10, r=15),
            yaxis={"showgrid": True},
            xaxis={"showgrid": True},
            coloraxis=dict(cmax=10, cmin=-10),
        )

        fig.update_xaxes(
            title_text="PCM comp-1",
            title_font={"size": 20},
            title_standoff=25,
            tickfont={"size": 15},
        )

        fig.update_yaxes(
            title_text="PCM comp-2",
            title_font={"size": 20},
            title_standoff=25,
            tickfont={"size": 15},
        )
        fig.update_traces(
            marker=dict(line=dict(width=2, color="grey")),
            selector=dict(mode="markers"),
        )

        cluster = dbc.Container(
            [html.H4("Video Semantic Clustering"), dcc.Graph(figure=fig)],
            className="m-5",
        )
    else:
        # PCA analysis has no sense w/ one video only.
        cluster = dbc.Container(
            [
                html.H4("Insufficient data to compare"),
            ],
            className="m-5",
        )

    fig2 = px.line(
        dff.reset_index(),
        x="index",
        y="sentiment_avg",
        hover_data=["videoTitle"],
        markers=True,
    )

    fig2.update_layout(
        showlegend=False,
        margin=dict(t=10, l=10, b=10, r=15),
        yaxis={"showgrid": True, "range": [-11, 11]},
        xaxis={"showgrid": True},
    )
    fig2.update_traces(marker={"size": 20, "color": "red"}, line={"color": "red"})
    fig2.update_xaxes(
        title_text="Video Index",
        title_font={"size": 20},
        title_standoff=25,
        type="category",
        tickfont={"size": 15},
    )

    fig2.update_yaxes(
        title_text="Avg Sentiment Score",
        title_font={"size": 20},
        title_standoff=25,
        tickfont={"size": 15},
    )
    fig2.add_hline(y=0, line_width=3, line_dash="dash", line_color="grey")

    sentiment = dbc.Container(
        [html.H4("Video Comments Sentiment Score"), dcc.Graph(figure=fig2)],
        className="m-5",
    )

    return cluster, sentiment


# main callback. Perform the video search, build the dataset with all video info.
# call Openai model to summarize the transcript, to eval the comment sentiment and the transcript embeddings.
# perform the PCA analysis on the embedding.
# generate the plots.
# IMPORTANT: during the video search only the video with comments and trascript are collected. The other discarted.
# this is the reson to define MAX results count. The resulting video could be less.
# POSSIBLE improvements: code could be improved to continue the search till the max results count is met.
# IMPORTANT: to save time the comment collection stop at 10 but the limit could be removed from the code and collect all comments.
# IMPORTANT: now the video duration you can force to short.  You can select another options.
@callback(
    Output("crossfilter-example", "rowData"),
    Output("memory-output", "data"),
    Output("id-cluster", "children", allow_duplicate=True),
    Output("id-sentiment", "children", allow_duplicate=True),
    Input("btn-search", "n_clicks"),
    State("input-topic", "value"),
    State("input-count", "value"),
    State("input-date", "value"),
    State("radios", "value"),
    prevent_initial_call=True,
)
def sth(n, topic, res_count, pub_date, duration):
    print("run YT query")
    if n != None:
        print(topic)
        print(topic)
        if duration == 1:
            videoDuration = "short"
        else:
            videoDuration = "medium"
        request = youtube.search().list(
            part="snippet",
            maxResults=res_count,
            q=topic,
            order="viewCount",
            publishedAfter=f"{pub_date}T00:00:00Z",
            relevanceLanguage="en",
            type="video",
            videoDuration=videoDuration,  # "short",
        )
        response = request.execute()

        # extract info from the results to build the dataset
        print(f"video duration {videoDuration}")

        videoId = []
        videoTitle = []
        videoTimeStamp = []
        videoDescription = []
        videoPic = []
        videoDict = {
            "videoId": videoId,
            "videoTitle": videoTitle,
            "videoTimeStamp": videoTimeStamp,
            "videoDescription": videoDescription,
            "videoPic": videoPic,
        }

        for i in response["items"]:
            videoId.append(i["id"]["videoId"])
            videoTitle.append(i["snippet"]["title"])
            videoTimeStamp.append(i["snippet"]["publishedAt"])
            videoDescription.append(i["snippet"]["description"])
            videoPic.append(i["snippet"]["thumbnails"]["medium"]["url"])

        df = pd.DataFrame(videoDict)

        comments = []
        tokens = []
        transcripts = []
        views = []
        duration = []
        author = []
        transcripts_ntokens = []
        # call custom separated function to get comments, transcripts and assess the nTokens needed for the AI requests
        for videoId in df.videoId.values:
            video_comments, comments_token = get_comments(youtube, videoId)
            comments.append(video_comments)
            tokens.append(comments_token)
            video_transcript, view_count, video_author, video_length, ntokens = (
                get_video_info(videoId)
            )
            transcripts.append(video_transcript)
            views.append(view_count)
            author.append(video_author)
            duration.append(video_length)
            transcripts_ntokens.append(ntokens)
        df["comments"] = comments
        df["commentsToken"] = tokens
        df["transcript"] = transcripts
        df["transcriptToken"] = transcripts_ntokens
        df["view"] = views
        df["author"] = author
        df["duration"] = duration
        # print(df.comments)

        # set token limits to avoid too long (expensive) trascripts passed to the AI
        df = (
            df[
                (df.transcript != "INVALID")
                & (df.transcriptToken < 3500)
                & (df.comments != "INVALID")
            ]
            .reset_index()
            .drop(columns="index")
        )
        df["videoLink"] = df.videoId.apply(
            lambda x: f"https://www.youtube.com/watch?v={x}"
        )
        df["videoTimeStamp"] = df.videoTimeStamp.str.replace(
            "T[0-9]*.*$", "", regex=True
        )

        # first AI features - transcrips summarization
        print("transcripts summarization")
        llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert at summarizing video transcripts to help users understand the value of the video about: {query_topic} by summarizing the trascript:{transcript}"
                    "Follow the user's indications when summarizing the video.",
                ),
                ("human", "please summarize the video in {max_word} maximum words"),
            ]
        )
        output_parser = StrOutputParser()

        chain = prompt | llm | output_parser

        response = []
        query_topic = topic
        max_word = 150
        for i in range(df.shape[0]):
            transcript = df["transcript"].loc[i]

            res = chain.invoke(
                {
                    "query_topic": query_topic,
                    "transcript": transcript,
                    "max_word": max_word,
                }
            )
            response.append(res)
        df["summary"] = response

        # second AI feature - sentiment analysis
        print("sentiment analysis")
        template = """
        Identify the sentiment towards the video comment {comment} from -10 to +10 where -10 being the most negative and +10 being the most positve , and 0 being neutral.
        GIVE ANSWER IN ONLY ONE WORD AND THAT SHOULD BE THE SCORE"""

        # forming prompt using Langchain PromptTemplate functionality
        prompt_sentiment = PromptTemplate(
            template=template, input_variables=["comment"]
        )
        llm_sentiment = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)
        output_parser_sentiment = StrOutputParser()
        chain_sentiment = prompt_sentiment | llm_sentiment | output_parser_sentiment
        df["sentiment_avg"] = 0

        for v in range(df.shape[0]):
            response_sentiment = []
            comment_list = df.comments.loc[v]
            for i in range(len(comment_list)):
                comment = comment_list[i]
                res = chain_sentiment.invoke({"comment": comment})
                response_sentiment.append(float(res))
            df.sentiment_avg.loc[v] = np.mean(response_sentiment)
        print("sentiment")
        df["sentiment_avg"].fillna(value=0, inplace=True)

        # third AI feature - embeddings
        print("sembeddings")
        embeddings = OpenAIEmbeddings()
        transcript_list = df.transcript.values.tolist()
        transcript_embed = []
        # run PCA for semantic clustering - only with more than 1 video.
        if len(transcript_list) > 1:
            for tr in transcript_list:
                query_result = embeddings.embed_query(tr)
                print("embedding eval")
                transcript_embed.append(query_result)
            df["transcript_embed"] = transcript_embed

            pca = PCA(n_components=2)
            df["transcript_embed_pca"] = pca.fit_transform(
                df.transcript_embed.values.tolist()
            ).tolist()
            pca_x = []
            pca_y = []

            for v in df["transcript_embed_pca"].values:
                pca_x.append(v[0])
                pca_y.append(v[1])
            df["transcript_embed_pca_x"] = pca_x
            df["transcript_embed_pca_y"] = pca_y

            # support variable to control the scatter marker size. NO physical meanings
            df["view_resize"] = 30

        cluster = dbc.Container(
            [
                html.H4("Run Analysis"),
            ],
            className="m-5",
        )

        sentiment = dbc.Container(
            [
                html.H4("Run Analysis"),
            ],
            className="m-5",
        )

        print("DONE")

        return df.to_dict("records"), df.to_dict("records"), cluster, sentiment
    else:
        dash.exceptions.PreventUpdate()
        sentiment = dbc.Container(
            [],
            className="m-5",
        )
        cluster = dbc.Container(
            [],
            className="m-5",
        )
        # return df.to_dict("records"), df.to_dict("records"), cluster, sentiment

    return df.to_dict("records"), df.to_dict("records"), cluster, sentiment


# populate the video card details. Triggered by selecting intem in ag-grid
@callback(
    Output("id-video-card", "children"),
    Input("crossfilter-example", "selectedRows"),
    State("memory-output", "data"),
    # prevent_initial_call=True,
)
def s(selected, data):
    dff_f = df_default
    if data is None:
        dff_f = df_default
    else:
        dff = pd.DataFrame(data)

    selected = [s["videoTitle"] for s in selected] if selected else []
    videoPic_default = "https://static.vecteezy.com/system/resources/previews/023/986/473/original/youtube-logo-youtube-logo-transparent-youtube-icon-transparent-free-free-png.png"

    card = card = create_videoCard_default(
        videoPic_default,
        "No results. Setup and run your search",
    )
    if len(selected) == 0:
        card = create_videoCard_default(
            videoPic_default,
            "No results. Setup and run your search",
        )
    else:
        if selected[0].startswith("No results"):
            card = create_videoCard_default(
                videoPic_default,
                "No results. Setup and run your search",
            )
        else:
            dff_f = dff[dff.videoTitle == selected[0]].reset_index()
            videoPic = dff_f.videoPic.loc[0]
            card = create_videoCard(videoPic, dff_f["videoTitle"].loc[0], dff_f)
    return card
