import os
import json
import time
import xml.etree.ElementTree as ET
import requests
import tempfile
import subprocess
import re
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
# DOWNLOAD AUDIO
# ─────────────────────────────────────────
def download_audio(video_url):
    """
    Uses yt-dlp to download only the audio from a YouTube video.
    Returns the path to the downloaded audio file.
    """
    tmp_dir = tempfile.mkdtemp()
    output_template = os.path.join(tmp_dir, "audio.%(ext)s")

    command = [
        "yt-dlp",
        "--no-playlist",
        "-x",                          # extract audio only
        "--audio-format", "wav",       # Groq Whisper works best with wav
        "--audio-quality", "0",        # best quality
        "-o", output_template,
        video_url
    ]

    print(f"  ⬇️  Downloading audio for: {video_url}")
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ❌ yt-dlp error: {result.stderr}")
        return None

    # Find the actual output file (extension may vary)
    for file in os.listdir(tmp_dir):
        if file.startswith("audio."):
            return os.path.join(tmp_dir, file)

    return None


# ─────────────────────────────────────────
# TRANSCRIBE WITH GROQ WHISPER
# ─────────────────────────────────────────
def transcribe_audio(audio_path):
    """
    Sends the audio file to Groq's Whisper endpoint and returns the transcript text.
    """
    client = Groq(api_key=GROQ_API_KEY)

    print(f"  🎧 Transcribing audio...")
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=("audio.wav", audio_file),
            model="whisper-large-v3-turbo",
            language="en"
        )

    return transcription.text


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
# CLEANUP TEMP FILES
# ─────────────────────────────────────────
def cleanup(audio_path):
    """Removes the temporary audio file after processing."""
    if audio_path and os.path.exists(audio_path):
        tmp_dir = os.path.dirname(audio_path)
        os.remove(audio_path)
        os.rmdir(tmp_dir)


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

            audio_path = None
            try:
                # Step 1: Download audio
                audio_path = download_audio(video["url"])
                if not audio_path:
                    print("  ⚠️  Skipping — audio download failed.")
                    seen.append(video["id"])
                    continue

                # Step 2: Transcribe
                transcript = transcribe_audio(audio_path)
                if not transcript or len(transcript.strip()) < 50:
                    print("  ⚠️  Skipping — transcript too short or empty.")
                    seen.append(video["id"])
                    continue

                # Step 3: Summarize
                summary = summarize_transcript(transcript, video["title"])
                video["summary"] = summary

                # Step 4: Send Telegram message
                caption = format_telegram_caption(video)
                send_telegram_photo(video["thumbnail"], caption)

                # Mark as seen
                seen.append(video["id"])

            except Exception as e:
                print(f"  ❌ Error processing video: {e}")
                # Still mark as seen so we don't retry and get stuck
                seen.append(video["id"])

            finally:
                cleanup(audio_path)

    # Save updated seen list
    save_seen_videos(seen)
    print("\n✅ Check complete. Seen videos saved.")


if __name__ == "__main__":
    main()
