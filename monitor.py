import os
import json
import xml.etree.ElementTree as ET
import requests
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
# GET TRANSCRIPT VIA YOUTUBETOTRANSCRIPT
# ─────────────────────────────────────────
def get_transcript(video_id):
    """
    Fetches the transcript by scraping youtubetotranscript.com.
    This avoids YouTube's cloud IP blocking entirely.
    """
    print(f"  📝 Fetching transcript for video: {video_id}")

    url = "https://youtubetotranscript.com/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://youtubetotranscript.com/"
    }

    # Step 1: Get the main page to grab any session cookies
    session = requests.Session()
    session.get(url, headers=headers)

    # Step 2: Submit the video URL to get the transcript
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    data = {"youtube_url": youtube_url}

    response = session.post(url, headers=headers, data=data)

    if response.status_code != 200:
        print(f"  ❌ Failed to fetch transcript: HTTP {response.status_code}")
        return None

    # Step 3: Extract the transcript text from the HTML response
    # The transcript is inside a <div> with id="transcript_text" or similar
    html = response.text

    # Try to find transcript in common patterns
    # Pattern 1: inside a textarea or div with specific id
    patterns = [
        r'<textarea[^>]*id=["\']transcript["\'][^>]*>(.*?)</textarea>',
        r'<div[^>]*id=["\']transcript_text["\'][^>]*>(.*?)</div>',
        r'<div[^>]*class=["\']transcript["\'][^>]*>(.*?)</div>',
        r'<div[^>]*id=["\']output["\'][^>]*>(.*?)</div>',
    ]

    transcript_text = None
    for pattern in patterns:
        match = re.search(pattern, html, re.DOTALL)
        if match:
            transcript_text = match.group(1)
            break

    # If none of the patterns matched, try a broader approach
    # Look for large blocks of text that look like a transcript
    if not transcript_text:
        # Try finding text between specific markers
        match = re.search(r'class="[^"]*transcript[^"]*"[^>]*>(.*?)</div>', html, re.DOTALL | re.IGNORECASE)
        if match:
            transcript_text = match.group(1)

    if not transcript_text:
        print("  ⚠️  Could not extract transcript from response.")
        # Save the HTML for debugging so we can see what we got back
        with open("debug_response.html", "w") as f:
            f.write(html)
        print("  📄 Saved debug_response.html for inspection.")
        return None

    # Clean up HTML tags and decode entities
    transcript_text = re.sub(r'<[^>]+>', ' ', transcript_text)
    transcript_text = transcript_text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
    transcript_text = re.sub(r'\s+', ' ', transcript_text).strip()

    if len(transcript_text) < 50:
        print("  ⚠️  Transcript is too short — video might not have captions.")
        return None

    print(f"  ✅ Got transcript ({len(transcript_text)} characters)")
    return transcript_text


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
                # Step 1: Get transcript
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
