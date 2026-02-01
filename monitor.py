import os
import json
import xml.etree.ElementTree as ET
import requests
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptNotAvailable, VideoUnplayable, NoTranscriptFound
from groq import Groq

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID  = os.environ["TELEGRAM_CHAT_ID"]
GROQ_API_KEY      = os.environ["GROQ_API_KEY"]

# Add more RSS feeds here if you want to monitor multiple channels.
# Format: "Friendly Name": "RSS URL"
YOUTUBE_CHANNELS = {
    "Hormozi Highlights": "https://www.youtube.com/feeds/videos.xml?channel_id=UCrvchO1h6lWZAuGaa1LqX9Q"
}

# File that stores which video IDs we've already processed
SEEN_VIDEOS_FILE = "seen_videos.json"


# ─────────────────────────────────────────
# LOAD / SAVE SEEN VIDEOS
# ─────────────────────────────────────────
def load_seen_videos():
    """Loads previously seen video IDs from the JSON file."""
    if os.path.exists(SEEN_VIDEOS_FILE):
        with open(SEEN_VIDEOS_FILE, "r") as f:
            return json.load(f)
    return []

def save_seen_videos(seen):
    """Saves the list of seen video IDs to the JSON file."""
    with open(SEEN_VIDEOS_FILE, "w") as f:
        json.dump(seen, f, indent=2)


# ─────────────────────────────────────────
# PARSE RSS FEED
# ─────────────────────────────────────────
def get_latest_videos(rss_url):
    """
    Fetches the YouTube RSS feed and returns a list of videos.
    Each video is a dict with: id, title, url, thumbnail
    """
    response = requests.get(rss_url)
    response.raise_for_status()

    root = ET.fromstring(response.content)

    # YouTube RSS uses the 'atom' namespace
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "media": "http://search.yahoo.com/mrss/"
    }

    videos = []
    for entry in root.findall("atom:entry", ns):
        video_id   = entry.find("atom:id", ns).text.split("/")[-1]  # extract ID from urn
        title      = entry.find("atom:title", ns).text
        link       = entry.find("atom:link", ns).attrib["href"]
        # Thumbnail is inside <media:group><media:thumbnail>
        media_group = entry.find("media:group", ns)
        thumbnail  = media_group.find("media:thumbnail", ns).attrib["url"] if media_group is not None else None

        videos.append({
            "id":        video_id,
            "title":     title,
            "url":       link,
            "thumbnail": thumbnail
        })

    return videos


# ─────────────────────────────────────────
# GET TRANSCRIPT FROM YOUTUBE CAPTIONS
# ─────────────────────────────────────────
def get_transcript(video_id):
    """
    Fetches the transcript directly from YouTube's auto-generated captions.
    No audio download needed — fast and doesn't trigger bot detection.
    """
    print(f"  📝 Fetching transcript...")
    try:
        # Try to get English transcript first, then fall back to any available
        transcript_list = YouTubeTranscriptApi.get_transcripts(
            [video_id],
            languages=["en", "en-US", "en-GB"]
        )
        # transcript_list is a tuple: (transcripts_dict, video_id)
        transcripts = transcript_list[0][video_id]

        # Pick the first available transcript
        if not transcripts:
            print("  ⚠️  No transcripts available for this video.")
            return None

        transcript_data = transcripts[0].fetch()

        # Combine all transcript chunks into one plain text string
        full_text = " ".join([chunk.text for chunk in transcript_data])
        return full_text

    except (NoTranscriptFound, TranscriptNotAvailable) as e:
        print(f"  ⚠️  No transcript found: {e}")
        return None
    except VideoUnplayable as e:
        print(f"  ⚠️  Video is unplayable: {e}")
        return None
    except Exception as e:
        print(f"  ⚠️  Transcript error: {e}")
        return None


# ─────────────────────────────────────────
# SUMMARIZE WITH GROQ LLM
# ─────────────────────────────────────────
def summarize_transcript(transcript, video_title):
    """
    Sends the transcript to Groq's LLM and returns a clean summary.
    """
    client = Groq(api_key=GROQ_API_KEY)

    prompt = f"""You are a helpful assistant that summarizes YouTube video transcripts.
Below is the transcript of a YouTube video titled: "{video_title}"

Please provide a clear, concise summary that:
- Captures the main topic and key points
- Is written in 3-5 bullet points
- Each bullet point starts with a relevant emoji (e.g. 💡 🔥 ⚡ 📌 🎯 🚀 💎 🛠️ 📊 ✅ etc.) chosen based on what that point is about
- Is easy to understand without watching the video
- Does NOT include any filler or generic statements
- Do NOT add any extra text before or after the bullet points, just the bullet points only

Transcript:
{transcript}

Summary:"""

    print(f"  🤖 Summarizing transcript...")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────
# SEND TELEGRAM MESSAGE
# ─────────────────────────────────────────
def send_telegram_photo(thumbnail_url, caption):
    """
    Sends a photo (thumbnail) with a caption to Telegram.
    The caption includes the summary and video link.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

    payload = {
        "chat_id":                TELEGRAM_CHAT_ID,
        "photo":                  thumbnail_url,
        "caption":                caption,
        "parse_mode":             "Markdown",
        "disable_notification":   False
    }

    print(f"  📱 Sending Telegram message...")
    response = requests.post(url, json=payload)

    if response.status_code != 200:
        print(f"  ❌ Telegram error: {response.text}")
        # Fallback: if photo fails, send as text message instead
        send_telegram_text(caption)
    else:
        print(f"  ✅ Telegram message sent successfully!")


def send_telegram_text(text):
    """Fallback: sends a plain text message if the photo fails."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text,
        "parse_mode": "Markdown"
    }
    requests.post(url, json=payload)


def format_telegram_caption(video):
    """
    Formats the Telegram caption with title, summary, link, and channel hashtag.
    """
    # Convert channel name to a hashtag (remove spaces, no special chars)
    hashtag = "#" + video["channel_name"].replace(" ", "")

    caption = (
        f"🎬 *{video['title']}*\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{video['summary']}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"🔗 [Watch Video]({video['url']})\n\n"
        f"{hashtag}"
    )
    return caption


# ─────────────────────────────────────────
# MAIN LOGIC
# ─────────────────────────────────────────
def main():
    print("🔍 YouTube Video Monitor started...")
    seen = load_seen_videos()

    for channel_name, rss_url in YOUTUBE_CHANNELS.items():
        print(f"\n📡 Checking channel: {channel_name}")

        try:
            videos = get_latest_videos(rss_url)
        except Exception as e:
            print(f"  ❌ Failed to fetch RSS feed: {e}")
            continue

        for video in videos:
            # Skip if we've already processed this video
            if video["id"] in seen:
                continue

            video["channel_name"] = channel_name
            print(f"\n🆕 New video detected: {video['title']}")

            try:
                # Step 1: Get transcript from YouTube captions
                transcript = get_transcript(video["id"])
                if not transcript or len(transcript.strip()) < 50:
                    print("  ⚠️  Skipping — transcript too short or not available.")
                    seen.append(video["id"])
                    continue

                # Step 2: Summarize
                summary = summarize_transcript(transcript, video["title"])
                video["summary"] = summary

                # Step 3: Send Telegram message
                caption = format_telegram_caption(video)
                send_telegram_photo(video["thumbnail"], caption)

                # Mark as seen
                seen.append(video["id"])

            except Exception as e:
                print(f"  ❌ Error processing video: {e}")
                # Still mark as seen so we don't retry and get stuck
                seen.append(video["id"])

    # Save updated seen list
    save_seen_videos(seen)
    print("\n✅ Check complete. Seen videos saved.")


if __name__ == "__main__":
    main()
