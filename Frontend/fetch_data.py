import os
import requests
from typing import List

# -------------------------------
# Twitter / X comments
# ------------------------------


# -------------------------------
# Reddit comments (via public .json endpoint)
# -------------------------------
# def fetch_reddit_comments(post_url: str, max_comments: int = 100) -> List[str]:
#     if "/comments/" not in post_url:
#         print("Error: Reddit URL must be a post (submission) link.")
#         return []

#     if not post_url.endswith(".json"):
#         post_url = post_url.rstrip("/") + ".json"

#     headers = {
#         "User-Agent": "sentiment-dashboard-bot/0.1 (educational project)"
#     }

#     try:
#         response = requests.get(post_url, headers=headers, timeout=10)
#         response.raise_for_status()
#         data = response.json()

#         comments = []

#         # data[1] contains comments
#         for item in data[1]["data"]["children"]:
#             if item["kind"] == "t1":
#                 body = item["data"].get("body")
#                 if body:
#                     comments.append(body)

#                 if len(comments) >= max_comments:
#                     break

#         return comments

#     except Exception as e:
#         print(f"Error fetching Reddit comments: {e}")
#         return []

from typing import List
from urllib.parse import urlparse
import requests

def fetch_reddit_comments(post_url: str, max_comments: int = 200) -> List[str]:
    headers = {
        "User-Agent": "sentiment-dashboard-bot/0.1 (educational project)"
    }

    comments = []

    def extract_comments(children):
        nonlocal comments

        for child in children:
            if len(comments) >= max_comments:
                return

            if child["kind"] != "t1":
                continue

            data = child["data"]
            body = data.get("body")

            if body:
                comments.append(body)

            # 🔁 Recursive call for replies
            replies = data.get("replies")
            if isinstance(replies, dict):
                extract_comments(replies["data"]["children"])

    try:
        # 🔹 Clean URL
        parsed = urlparse(post_url)
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        if "/comments/" not in clean_url:
            print("Invalid Reddit post URL")
            return []

        if not clean_url.endswith(".json"):
            clean_url = clean_url.rstrip("/") + ".json"

        resp = requests.get(clean_url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # 🔹 Start recursion from top-level comments
        extract_comments(data[1]["data"]["children"])

        return comments

    except Exception as e:
        print(f"Reddit fetch error: {e}")
        return []


# -------------------------------
# YouTube video comments (NO API KEY, FREE)
# -------------------------------
def fetch_video_comments(video_url: str, max_comments: int = 100) -> List[str]:
    try:
        from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
    except ImportError:
        print("youtube-comment-downloader not installed.")
        return ["Install youtube-comment-downloader to fetch real comments."]

    downloader = YoutubeCommentDownloader()
    comments = []

    try:
        for comment in downloader.get_comments_from_url(
            video_url,
            sort_by=SORT_BY_POPULAR
        ):
            text = comment.get("text")
            if text:
                comments.append(text)

            if len(comments) >= max_comments:
                break

        if not comments:
            return ["No comments found or comments are disabled."]

        return comments

    except Exception as e:
        print(f"Error fetching YouTube comments: {e}")
        return ["Failed to fetch YouTube comments."]


# -------------------------------
# Unified fetch function
# -------------------------------
def get_comments(source: str, source_type: str) -> List[str]:
    """
    Unified comment fetcher.
    source_type: 'tweet' | 'reddit' | 'video'
    """

    source_type = source_type.lower()

    if source_type == "reddit":
        return fetch_reddit_comments(source)

    elif source_type == "video":
        return fetch_video_comments(source)

    else:
        print(f"Warning: Unsupported source type '{source_type}'")
        return []
