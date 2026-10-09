from flask import Flask, render_template, request, redirect, Response
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import torch
import sqlite3
import csv
import io
import os
import re
import requests


app = Flask(__name__)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "./model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)


# ============================================================
# SENTIMENT LABELS
# ============================================================

LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


# ============================================================
# DATABASE
# ============================================================

DATABASE = "sentiment_history.db"


def get_db_connection():

    connection = sqlite3.connect(DATABASE)

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# CREATE DATABASE
# ============================================================

def create_database():

    connection = get_db_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_history (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            comment TEXT NOT NULL,

            sentiment TEXT NOT NULL,

            confidence REAL NOT NULL,

            negative_score REAL NOT NULL,

            neutral_score REAL NOT NULL,

            positive_score REAL NOT NULL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )
    """)

    connection.commit()

    connection.close()


# ============================================================
# PREDICT SENTIMENT
# ============================================================

def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    prediction = torch.argmax(
        probabilities
    ).item()

    sentiment = LABELS[prediction]

    confidence = (
        probabilities[prediction].item()
        * 100
    )

    scores = {

        "Negative": round(
            probabilities[0].item() * 100,
            2
        ),

        "Neutral": round(
            probabilities[1].item() * 100,
            2
        ),

        "Positive": round(
            probabilities[2].item() * 100,
            2
        )

    }

    return sentiment, confidence, scores


# ============================================================
# SAVE PREDICTION
# ============================================================

def save_prediction(
    comment,
    sentiment,
    confidence,
    scores
):

    connection = get_db_connection()

    connection.execute(
        """
        INSERT INTO sentiment_history
        (
            comment,
            sentiment,
            confidence,
            negative_score,
            neutral_score,
            positive_score
        )

        VALUES (?, ?, ?, ?, ?, ?)
        """,

        (
            comment,
            sentiment,
            round(confidence, 2),
            scores["Negative"],
            scores["Neutral"],
            scores["Positive"]
        )
    )

    connection.commit()

    connection.close()


# ============================================================
# GET HISTORY
# ============================================================

def get_history():

    connection = get_db_connection()

    history = connection.execute(
        """
        SELECT *

        FROM sentiment_history

        ORDER BY id DESC

        LIMIT 10
        """
    ).fetchall()

    connection.close()

    return history


# ============================================================
# GET DASHBOARD STATISTICS
# ============================================================

def get_statistics():

    connection = get_db_connection()


    total = connection.execute(
        """
        SELECT COUNT(*)
        FROM sentiment_history
        """
    ).fetchone()[0]


    positive = connection.execute(
        """
        SELECT COUNT(*)
        FROM sentiment_history
        WHERE sentiment = 'Positive'
        """
    ).fetchone()[0]


    neutral = connection.execute(
        """
        SELECT COUNT(*)
        FROM sentiment_history
        WHERE sentiment = 'Neutral'
        """
    ).fetchone()[0]


    negative = connection.execute(
        """
        SELECT COUNT(*)
        FROM sentiment_history
        WHERE sentiment = 'Negative'
        """
    ).fetchone()[0]


    connection.close()


    if total > 0:

        positive_percentage = round(
            (positive / total) * 100,
            2
        )

        neutral_percentage = round(
            (neutral / total) * 100,
            2
        )

        negative_percentage = round(
            (negative / total) * 100,
            2
        )

    else:

        positive_percentage = 0

        neutral_percentage = 0

        negative_percentage = 0


    return {

        "total": total,

        "positive": positive,

        "neutral": neutral,

        "negative": negative,

        "positive_percentage": positive_percentage,

        "neutral_percentage": neutral_percentage,

        "negative_percentage": negative_percentage

    }


# ============================================================
# EXTRACT YOUTUBE VIDEO ID
# ============================================================

def extract_video_id(url):

    patterns = [

        r"(?:v=)([0-9A-Za-z_-]{11})",

        r"(?:youtu\.be/)([0-9A-Za-z_-]{11})",

        r"(?:youtube\.com/shorts/)([0-9A-Za-z_-]{11})",

        r"(?:youtube\.com/embed/)([0-9A-Za-z_-]{11})",

        r"(?:youtube\.com/live/)([0-9A-Za-z_-]{11})"

    ]


    for pattern in patterns:

        match = re.search(
            pattern,
            url
        )

        if match:

            return match.group(1)


    return None


# ============================================================
# GET YOUTUBE VIDEO INFORMATION
# ============================================================

def get_video_information(video_id):

    if not YOUTUBE_API_KEY:

        return None, "YouTube API key was not found."


    url = "https://www.googleapis.com/youtube/v3/videos"


    params = {

        "part": "snippet",

        "id": video_id,

        "key": YOUTUBE_API_KEY

    }


    try:

        response = requests.get(
            url,
            params=params,
            timeout=20
        )


        data = response.json()


        if response.status_code != 200:

            error_message = data.get(
                "error",
                {}
            ).get(
                "message",
                "Unable to retrieve video information."
            )

            return None, error_message


        items = data.get(
            "items",
            []
        )


        if not items:

            return None, "YouTube video was not found."


        snippet = items[0]["snippet"]


        video_info = {

            "title": snippet.get(
                "title",
                "Unknown"
            ),

            "channel": snippet.get(
                "channelTitle",
                "Unknown"
            ),

            "published": snippet.get(
                "publishedAt",
                "Unknown"
            ),

            "description": snippet.get(
                "description",
                ""
            )

        }


        return video_info, None


    except requests.RequestException as error:

        return None, (
            "Could not connect to YouTube API: "
            + str(error)
        )


# ============================================================
# GET YOUTUBE COMMENTS
# ============================================================

def get_youtube_comments(
    video_id,
    max_comments=50
):

    if not YOUTUBE_API_KEY:

        return [], "YouTube API key was not found."


    comments = []

    next_page_token = None


    while len(comments) < max_comments:

        remaining = max_comments - len(comments)

        request_size = min(
            remaining,
            100
        )


        url = (
            "https://www.googleapis.com/"
            "youtube/v3/commentThreads"
        )


        params = {

            "part": "snippet",

            "videoId": video_id,

            "maxResults": request_size,

            "textFormat": "plainText",

            "order": "relevance",

            "key": YOUTUBE_API_KEY

        }


        if next_page_token:

            params["pageToken"] = next_page_token


        try:

            response = requests.get(
                url,
                params=params,
                timeout=20
            )


            data = response.json()


            if response.status_code != 200:

                error_message = data.get(
                    "error",
                    {}
                ).get(
                    "message",
                    "Unable to retrieve YouTube comments."
                )

                return comments, error_message


            items = data.get(
                "items",
                []
            )


            if not items:

                break


            for item in items:

                try:

                    comment = (
                        item["snippet"]
                        ["topLevelComment"]
                        ["snippet"]
                        ["textDisplay"]
                    )

                    if comment.strip():

                        comments.append(
                            comment.strip()
                        )


                except KeyError:

                    continue


                if len(comments) >= max_comments:

                    break


            next_page_token = data.get(
                "nextPageToken"
            )


            if not next_page_token:

                break


        except requests.RequestException as error:

            return comments, (
                "Could not connect to YouTube API: "
                + str(error)
            )


    return comments, None


# ============================================================
# ANALYZE YOUTUBE VIDEO
# ============================================================

def analyze_youtube_video(
    video_url,
    max_comments
):

    video_id = extract_video_id(
        video_url
    )


    if not video_id:

        return {

            "success": False,

            "error": (
                "Invalid YouTube URL. "
                "Please enter a valid YouTube video URL."
            )

        }


    video_info, video_error = get_video_information(
        video_id
    )


    if video_error:

        return {

            "success": False,

            "error": video_error

        }


    comments, comment_error = get_youtube_comments(
        video_id,
        max_comments
    )


    if comment_error:

        return {

            "success": False,

            "error": comment_error,

            "video_info": video_info

        }


    if not comments:

        return {

            "success": False,

            "error": (
                "No public comments were found "
                "for this video."
            ),

            "video_info": video_info

        }


    results = []


    positive = 0

    neutral = 0

    negative = 0


    for comment in comments:

        sentiment, confidence, scores = predict_sentiment(
            comment
        )


        save_prediction(
            comment,
            sentiment,
            confidence,
            scores
        )


        if sentiment == "Positive":

            positive += 1

        elif sentiment == "Neutral":

            neutral += 1

        else:

            negative += 1


        results.append({

            "comment": comment,

            "sentiment": sentiment,

            "confidence": round(
                confidence,
                2
            )

        })


    total = len(results)


    positive_percentage = round(
        (positive / total) * 100,
        2
    )

    neutral_percentage = round(
        (neutral / total) * 100,
        2
    )

    negative_percentage = round(
        (negative / total) * 100,
        2
    )


    return {

        "success": True,

        "video_id": video_id,

        "video_info": video_info,

        "comments_analyzed": total,

        "positive": positive,

        "neutral": neutral,

        "negative": negative,

        "positive_percentage": positive_percentage,

        "neutral_percentage": neutral_percentage,

        "negative_percentage": negative_percentage,

        "results": results

    }


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    sentiment = None

    confidence = None

    scores = None

    comment = ""

    batch_results = []

    youtube_result = None

    youtube_error = None


    if request.method == "POST":


        # ====================================================
        # SINGLE COMMENT
        # ====================================================

        if request.form.get(
            "form_type"
        ) == "single":

            comment = request.form.get(
                "comment",
                ""
            ).strip()


            if comment:

                sentiment, confidence, scores = predict_sentiment(
                    comment
                )


                save_prediction(
                    comment,
                    sentiment,
                    confidence,
                    scores
                )


        # ====================================================
        # BATCH COMMENTS
        # ====================================================

        elif request.form.get(
            "form_type"
        ) == "batch":

            batch_text = request.form.get(
                "batch_comments",
                ""
            ).strip()


            if batch_text:

                comments = batch_text.splitlines()


                for current_comment in comments:

                    current_comment = (
                        current_comment.strip()
                    )


                    if current_comment:

                        batch_sentiment, batch_confidence, batch_scores = predict_sentiment(
                            current_comment
                        )


                        save_prediction(
                            current_comment,
                            batch_sentiment,
                            batch_confidence,
                            batch_scores
                        )


                        batch_results.append({

                            "comment": current_comment,

                            "sentiment": batch_sentiment,

                            "confidence": round(
                                batch_confidence,
                                2
                            )

                        })


        # ====================================================
        # YOUTUBE VIDEO
        # ====================================================

        elif request.form.get(
            "form_type"
        ) == "youtube":

            video_url = request.form.get(
                "youtube_url",
                ""
            ).strip()


            try:

                max_comments = int(
                    request.form.get(
                        "max_comments",
                        50
                    )
                )

            except ValueError:

                max_comments = 50


            max_comments = max(
                1,
                min(
                    max_comments,
                    500
                )
            )


            youtube_result = analyze_youtube_video(
                video_url,
                max_comments
            )


            if not youtube_result.get(
                "success",
                False
            ):

                youtube_error = youtube_result.get(
                    "error",
                    "YouTube analysis failed."
                )


    history = get_history()

    statistics = get_statistics()


    return render_template(

        "index.html",

        sentiment=sentiment,

        confidence=confidence,

        scores=scores,

        comment=comment,

        history=history,

        statistics=statistics,

        batch_results=batch_results,

        youtube_result=youtube_result,

        youtube_error=youtube_error

    )


# ============================================================
# CLEAR ALL HISTORY
# ============================================================

@app.route(
    "/clear-history",
    methods=["POST"]
)
def clear_history():

    connection = get_db_connection()

    connection.execute(
        "DELETE FROM sentiment_history"
    )

    connection.commit()

    connection.close()

    return redirect("/")


# ============================================================
# EXPORT HISTORY TO CSV
# ============================================================

@app.route("/export-csv")
def export_csv():

    connection = get_db_connection()


    history = connection.execute(
        """
        SELECT
            id,
            comment,
            sentiment,
            confidence,
            negative_score,
            neutral_score,
            positive_score,
            created_at

        FROM sentiment_history

        ORDER BY id DESC
        """
    ).fetchall()


    connection.close()


    output = io.StringIO()

    writer = csv.writer(output)


    writer.writerow([

        "ID",

        "Comment",

        "Sentiment",

        "Confidence (%)",

        "Negative Score (%)",

        "Neutral Score (%)",

        "Positive Score (%)",

        "Date & Time"

    ])


    for row in history:

        writer.writerow([

            row["id"],

            row["comment"],

            row["sentiment"],

            row["confidence"],

            row["negative_score"],

            row["neutral_score"],

            row["positive_score"],

            row["created_at"]

        ])


    response = Response(
        output.getvalue(),
        mimetype="text/csv"
    )


    response.headers["Content-Disposition"] = (
        "attachment; filename=sentiment_history.csv"
    )


    return response


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    create_database()

    app.run(debug=True)