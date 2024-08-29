import tiktoken
from langchain_community.document_loaders import YoutubeLoader
import os
from dotenv import load_dotenv


# support functions to manage the env and retrieve video details and eval token counts
def getEnvVar():

    load_dotenv(override=True)
    API_KEY = os.environ.get("OPENAI_API_KEY")
    return API_KEY


def count_tokens(text, model):

    encoding = tiktoken.encoding_for_model(model)

    tokens = int(len(encoding.encode(text)))
    return tokens


# Function to get YouTube comments
def get_comments(youtube, video_id):

    comments = []
    commentsTokens = []
    try:
        response = (
            youtube.commentThreads()
            .list(
                part="snippet", videoId=video_id, textFormat="plainText", maxResults=5
            )
            .execute()
        )

        while response:
            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                numToken = count_tokens(comment, "gpt-4o")
                commentsTokens.append(int(numToken))
                comments.append(comment)

            # hardcoded 10 is to truncate the comment retrieval - it could be removed to get all comments
            if "nextPageToken" in response and len(comments) < 10:
                response = (
                    youtube.commentThreads()
                    .list(
                        part="snippet",
                        videoId=video_id,
                        textFormat="plainText",
                        pageToken=response["nextPageToken"],
                        maxResults=10,
                    )
                    .execute()
                )

            else:
                break
    except:
        comments.append("INVALID")
        commentsTokens.append(0)

    return comments, commentsTokens


def get_video_info(videoId):
    loader = YoutubeLoader.from_youtube_url(
        f"https://www.youtube.com/watch?v={videoId}", add_video_info=True
    )
    try:
        info = loader.load()

        print("VALID")
        print(videoId)
        transcript = info[0].page_content
        view = info[0].metadata["view_count"]
        author = info[0].metadata["author"]
        length = info[0].metadata["length"]
        transcript_tokens = count_tokens(transcript, "text-embedding-ada-002")
    except:
        print("EXCEPTION_INVALID")
        transcript = "INVALID"
        view = "INVALID"
        author = "INVALID"
        length = "INVALID"
        transcript_tokens = 0
    return transcript, view, author, length, transcript_tokens
