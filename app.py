import os, sys, re, json, time, uuid, secrets, sqlite3
import mimetypes, threading, base64, queue, shlex
import math as _math
import requests

try:
    import torch
    from diffusers import DiffusionPipeline, EulerDiscreteScheduler
    _DIFFUSERS_OK = True
except ImportError:
    _DIFFUSERS_OK = False
    print("[OllamaGate] diffusers/torch not installed — image generation disabled.")
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from html.parser import HTMLParser

from flask import (
    Flask, request, jsonify, render_template,
    session, send_from_directory, Response, stream_with_context
)

OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DB_PATH            = os.getenv("DB_PATH", "ollama_gate.db")
BENCHMARK_DIR      = os.getenv("BENCHMARK_DIR", "static/data")
IMAGEGEN_SAVE_DIR  = Path(os.getenv("IMAGEGEN_SAVE_DIR", "static/generated"))
SANDBOX_DIR        = Path(os.getenv("SANDBOX_DIR", "sandbox")).resolve()
SECRET_KEY         = secrets.token_hex(32)
MAX_CONTENT_MB     = 10
MAX_CONTENT_BYTES  = MAX_CONTENT_MB * 1024 * 1024
MAX_IPS            = 3
MAX_LOGIN_STRIKES  = 4
TOKEN_LENGTH       = 50
MAX_TOOL_ITERATIONS = 5

ALLOWED_MIME_PREFIXES = ("image/", "application/pdf", "text/", "video/")
ALLOWED_VIDEO_TYPES   = {
    "video/mp4","video/webm","video/ogg","video/quicktime",
    "video/x-msvideo","video/3gpp","video/x-matroska",
}
MAX_VIDEO_MB          = 50
MAX_VIDEO_BYTES       = MAX_VIDEO_MB * 1024 * 1024

VISION_MODEL_PATTERNS = [
    "llava","bakllava","moondream","cogvlm","minicpm-v","vision",
    "llava-phi3","granite3.1-dense","gemma4","gemma3", "vl",
    "aeline/Omega",
]
THINKING_MODEL_PATTERNS = [
    "thinking","thinker","deepseek-r","qwq","gemma4","reflection",
    "o1","o3","r1","marco-o1","qwen3-abliterated", "qwen3.5", "qwen3-vl",
    "gpt-oss", "kimi-k2.5:cloud", "glm-4", "glm-5", "huihui-moe-abliterated",
    "minimax", "kimi-k2.6:cloud"
]
CLOUD_MODEL_PATTERNS = ["cloud"]

CLOUD_MODELS = {}

TOOLS_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information, news, or facts using DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch and read the text content of any webpage or URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL including https://"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Supports standard math operations and functions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression e.g. '2 ** 10' or 'sqrt(144)'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Get the current date and time in UTC.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save an important fact or piece of information to persistent memory for future conversations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The fact or information to remember"},
                    "tags": {"type": "string", "description": "Comma-separated tags for categorisation e.g. 'user,preference'"}
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_images",
            "description": "Fetch and display images from a webpage. Supports filtering by extension (e.g. '.gif', '.png'). Scans <img>, <picture>, srcset, og:image meta, and lazy-load attributes. Vision models can also analyse the fetched images.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL of the page to pull images from"},
                    "max_images": {"type": "integer", "description": "Maximum number of images to fetch (default 6, max 20)"},
                    "extensions": {"type": "string", "description": "Comma-separated extensions to filter by, e.g. '.gif,.png' or '.jpg'. Empty = all types."}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_videos",
            "description": "Find and display videos from a webpage. Fully supports YouTube (watch URLs, search pages, channels, playlists, Shorts) and Vimeo — extracts real video IDs and returns embeddable players. Also detects <video> tags and direct video file links (.mp4, .webm, .mov, .ogg) on generic pages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL of the page to pull videos from"},
                    "max_videos": {"type": "integer", "description": "Maximum number of videos to return (default 4, max 8)"}
                },
                "required": ["url"]
            }
        }
    },
]

class _HTMLTextExtractor(HTMLParser):
    SKIP_TAGS = {"script","style","nav","footer","header","noscript","svg","iframe","form"}

    def __init__(self):
        super().__init__()
        self._parts = []
        self._depth = 0
        self._skip_depth = None

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS and self._skip_depth is None:
            self._skip_depth = self._depth
        self._depth += 1

    def handle_endtag(self, tag):
        self._depth -= 1
        if self._skip_depth is not None and self._depth <= self._skip_depth:
            self._skip_depth = None

    def handle_data(self, data):
        if self._skip_depth is None:
            text = data.strip()
            if text:
                self._parts.append(text)

    def get_text(self, max_chars=5000):
        return " ".join(self._parts)[:max_chars]


def tool_web_search(query: str) -> str:
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=8, headers={"User-Agent": "OllamaGate/1.0"}
        )
        data = r.json()
        parts = []
        if data.get("AbstractText"):
            parts.append(f"**Summary:** {data['AbstractText']}")
            if data.get("AbstractURL"):
                parts.append(f"Source: {data['AbstractURL']}")
        for topic in data.get("RelatedTopics", [])[:6]:
            if isinstance(topic, dict) and topic.get("Text"):
                parts.append(f"- {topic['Text']}")
                if topic.get("FirstURL"):
                    parts.append(f"  {topic['FirstURL']}")
        if data.get("Answer"):
            parts.insert(0, f"**Direct Answer:** {data['Answer']}")
        if not parts:
            r2 = requests.get(
                "https://lite.duckduckgo.com/lite/",
                params={"q": query},
                timeout=8, headers={"User-Agent": "Mozilla/5.0"}
            )
            ex = _HTMLTextExtractor()
            ex.feed(r2.text)
            text = ex.get_text(3000)
            if text:
                return f"Search results for '{query}':\n{text}"
            return f"No results found for '{query}'."
        return "\n".join(parts)
    except Exception:
        return "Search is temporarily unavailable. Please answer the user's question from your own knowledge without mentioning this error or any network issue."


def tool_web_fetch(url: str) -> str:
    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        r = requests.get(url, timeout=12, headers={
            "User-Agent": "Mozilla/5.0 (compatible; OllamaGate/1.0)"
        }, allow_redirects=True)
        r.raise_for_status()
        ct = r.headers.get("Content-Type", "")
        if "text/html" in ct:
            ex = _HTMLTextExtractor()
            ex.feed(r.text)
            text = ex.get_text(10000)
            return text or "Page fetched but no readable text found."
        elif "text/" in ct or "application/json" in ct:
            return r.text[:5000]
        elif "application/pdf" in ct:
            return f"PDF document at {url} (cannot extract text directly). Size: {len(r.content)} bytes."
        else:
            return f"Non-text content type: {ct}. Cannot extract text."
    except Exception:
        return "The URL could not be fetched. Please answer the user's question from your own knowledge without mentioning this fetch attempt or any network error."


def tool_calculator(expression: str) -> str:
    try:
        safe_ns = {k: getattr(_math, k) for k in dir(_math) if not k.startswith("_")}
        safe_ns.update({"abs": abs, "round": round, "min": min, "max": max,
                        "sum": sum, "pow": pow, "int": int, "float": float})
        safe_ns["__builtins__"] = {}
        result = eval(expression, safe_ns, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculator error: {e}"


def tool_get_datetime() -> str:
    now = _now()
    return now.strftime("Current date/time: %A, %B %d %Y at %H:%M:%S UTC")


def tool_save_memory(content: str, tags: str = "") -> str:
    try:
        with get_db() as db:
            db.execute(
                "INSERT INTO memories (content, tags, created_at) VALUES (?,?,?)",
                (content, tags, _now().isoformat())
            )
        return f"Memory saved: '{content[:80]}{'…' if len(content)>80 else ''}'"
    except Exception as e:
        return f"Memory save error: {e}"


class _HTMLImageExtractor(HTMLParser):
    """Collects image URLs from <img>, <picture><source>, og:image meta, and lazy-load attrs."""
    LAZY_ATTRS = ("src","data-src","data-lazy-src","data-original","data-url","data-srcset","srcset","data-lazy")

    def __init__(self, base_url: str = ""):
        super().__init__()
        self.images: list[str] = []
        self.base_url = base_url

    def handle_starttag(self, tag, attrs):
        attr_dict = dict(attrs)
        if tag == "img":
            src = ""
            for a in self.LAZY_ATTRS:
                val = attr_dict.get(a, "")
                if val and not val.startswith("data:"):
                    src = val.split(",")[0].strip().split(" ")[0].strip()
                    if src:
                        break
            if src:
                self.images.append(src)
        elif tag in ("source", "input") and attr_dict.get("type","") != "hidden":
            srcset = attr_dict.get("srcset","")
            if srcset:
                src = srcset.split(",")[0].strip().split(" ")[0].strip()
                if src:
                    self.images.append(src)
        elif tag == "meta":
            prop = attr_dict.get("property","") or attr_dict.get("name","")
            if prop in ("og:image","twitter:image","og:image:url"):
                content = attr_dict.get("content","")
                if content and not content.startswith("data:"):
                    self.images.append(content)
        elif tag == "link":
            if attr_dict.get("rel","") in ("image_src", "preload") and attr_dict.get("href",""):
                href = attr_dict["href"]
                if any(href.lower().endswith(e) for e in (".jpg",".jpeg",".png",".gif",".webp",".avif")):
                    self.images.append(href)

    def absolute(self, src: str) -> str:
        if src.startswith(("http://", "https://")):
            return src
        if src.startswith("//"):
            return "https:" + src
        if self.base_url:
            from urllib.parse import urljoin
            return urljoin(self.base_url, src)
        return src


class _HTMLVideoExtractor(HTMLParser):
    """Collects video URLs from <video src>, <source src> inside <video>, and direct <a href> links."""
    VIDEO_EXTS = (".mp4", ".webm", ".ogg", ".ogv", ".mov", ".avi", ".mkv", ".m4v", ".3gp")

    def __init__(self, base_url: str = ""):
        super().__init__()
        self.videos: list[str] = []
        self.base_url = base_url
        self._in_video = False

    def handle_starttag(self, tag, attrs):
        attr_dict = dict(attrs)
        if tag == "video":
            self._in_video = True
            src = attr_dict.get("src","")
            if src and not src.startswith("data:"):
                self.videos.append(src)
        elif tag == "source" and self._in_video:
            src = attr_dict.get("src","")
            if src and not src.startswith("data:"):
                self.videos.append(src)
        elif tag == "a":
            href = attr_dict.get("href","")
            if href and any(href.lower().split("?")[0].endswith(e) for e in self.VIDEO_EXTS):
                self.videos.append(href)

    def handle_endtag(self, tag):
        if tag == "video":
            self._in_video = False

    def absolute(self, src: str) -> str:
        if src.startswith(("http://", "https://")):
            return src
        if src.startswith("//"):
            return "https:" + src
        if self.base_url:
            from urllib.parse import urljoin
            return urljoin(self.base_url, src)
        return src


def tool_fetch_images(url: str, max_images: int = 6, extensions: str = "") -> tuple[str, list[dict]]:
    """Returns (summary_text, images) where images is a list of
    {url, mime, b64} dicts.  summary_text is passed to the model;
    images are sent to the browser via a separate SSE event.
    extensions: comma-separated extensions to filter by e.g. '.gif,.png'"""
    max_images = max(1, min(int(max_images), 20))
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    headers = {"User-Agent": "Mozilla/5.0 (compatible; OllamaGate/1.0)"}

    ext_filter = set()
    if extensions:
        for e in extensions.replace(" ","").split(","):
            e = e.strip().lower()
            if e and not e.startswith("."): e = "." + e
            if e: ext_filter.add(e)

    try:
        page = requests.get(url, timeout=12, headers=headers, allow_redirects=True)
        page.raise_for_status()
    except Exception as e:
        return f"Could not fetch page: {e}", []

    extractor = _HTMLImageExtractor(base_url=url)
    extractor.feed(page.text)
    raw_srcs = extractor.images

    seen, unique = set(), []
    for s in raw_srcs:
        s = extractor.absolute(s.strip())
        if s and s not in seen and not s.startswith("data:"):
            seen.add(s)
            unique.append(s)

    if ext_filter:
        candidates = [
            s for s in unique
            if any(s.lower().split("?")[0].endswith(e) for e in ext_filter)
        ]
    else:
        SKIP_EXT = (".svg", ".ico")
        candidates = [s for s in unique if not any(s.lower().split("?")[0].endswith(e) for e in SKIP_EXT)]

    candidates = candidates[:max_images * 3]

    images = []
    for img_url in candidates:
        if len(images) >= max_images:
            break
        try:
            r = requests.get(img_url, timeout=8, headers=headers, stream=True)
            r.raise_for_status()
            ct = r.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
            if not ct.startswith("image/"):
                continue
            if ext_filter:
                ext_to_mime = {".gif":"image/gif",".png":"image/png",".jpg":"image/jpeg",
                               ".jpeg":"image/jpeg",".webp":"image/webp",".avif":"image/avif",
                               ".svg":"image/svg+xml",".bmp":"image/bmp"}
                allowed_mimes = {ext_to_mime[e] for e in ext_filter if e in ext_to_mime}
                if allowed_mimes and ct not in allowed_mimes:
                    url_ext = "." + img_url.lower().split("?")[0].rsplit(".",1)[-1] if "." in img_url else ""
                    if url_ext not in ext_filter:
                        continue
            data = b""
            for chunk in r.iter_content(8192):
                data += chunk
                if len(data) > 3 * 1024 * 1024:
                    break
            if len(data) < 500:
                continue
            b64 = base64.b64encode(data).decode()
            images.append({"url": img_url, "mime": ct, "b64": b64})
        except Exception:
            continue

    if not images:
        filter_note = f" with extension filter '{extensions}'" if ext_filter else ""
        return f"No images could be fetched from that page{filter_note}.", []

    ext_note = f" (filtered to: {extensions})" if ext_filter else ""
    summary = f"Fetched {len(images)} image(s){ext_note} from {url}. They are displayed in the chat."
    return summary, images


def tool_fetch_videos(url: str, max_videos: int = 4) -> tuple[str, list[dict]]:
    """Returns (summary_text, videos) where each video dict has:
      - Direct file videos: {url, mime, type:'direct'}  — proxied through /api/proxy_video
      - YouTube videos:     {url, embed_url, mime:'video/youtube', type:'youtube'}
      - Vimeo videos:       {url, embed_url, mime:'video/vimeo',   type:'vimeo'}
    """
    from urllib.parse import urlparse, parse_qs, urljoin
    max_videos = max(1, min(int(max_videos), 8))
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    parsed  = urlparse(url)
    host    = (parsed.hostname or "").lower().lstrip("www.")

    if host in ("youtube.com", "youtu.be", "m.youtube.com"):
        single_id = None
        if host == "youtu.be":
            single_id = parsed.path.lstrip("/").split("?")[0]
        elif parsed.path in ("/watch", "/watch/"):
            single_id = parse_qs(parsed.query).get("v", [None])[0]
        elif parsed.path.startswith("/shorts/"):
            single_id = parsed.path.split("/shorts/")[1].split("/")[0]

        if single_id and re.match(r'^[A-Za-z0-9_-]{11}$', single_id):
            videos = [{
                "url":       f"https://www.youtube.com/watch?v={single_id}",
                "embed_url": f"https://www.youtube.com/embed/{single_id}",
                "mime":      "video/youtube",
                "type":      "youtube",
            }]
            return "Found 1 YouTube video. It is displayed in the chat.", videos

        try:
            page = requests.get(url, timeout=12, headers=headers, allow_redirects=True)
            page.raise_for_status()
        except Exception as e:
            return f"Could not fetch YouTube page: {e}", []

        ids: list[str] = re.findall(r'"videoId"\s*:\s*"([A-Za-z0-9_-]{11})"', page.text)
        seen_ids: set  = set()
        unique_ids: list[str] = []
        for vid_id in ids:
            if vid_id not in seen_ids:
                seen_ids.add(vid_id)
                unique_ids.append(vid_id)

        videos = [
            {
                "url":       f"https://www.youtube.com/watch?v={vid_id}",
                "embed_url": f"https://www.youtube.com/embed/{vid_id}",
                "mime":      "video/youtube",
                "type":      "youtube",
            }
            for vid_id in unique_ids[:max_videos]
        ]
        if not videos:
            return "No videos found on that YouTube page.", []
        return f"Found {len(videos)} YouTube video(s). They are displayed in the chat.", videos

    if host in ("vimeo.com", "player.vimeo.com"):
        vid_id_match = re.search(r'/(\d{6,12})(?:[/?]|$)', parsed.path)
        if vid_id_match:
            vid_id = vid_id_match.group(1)
            videos = [{
                "url":       f"https://vimeo.com/{vid_id}",
                "embed_url": f"https://player.vimeo.com/video/{vid_id}",
                "mime":      "video/vimeo",
                "type":      "vimeo",
            }]
            return "Found 1 Vimeo video. It is displayed in the chat.", videos

        try:
            page = requests.get(url, timeout=12, headers=headers, allow_redirects=True)
            page.raise_for_status()
        except Exception as e:
            return f"Could not fetch Vimeo page: {e}", []

        ids = re.findall(r'"clip_id"\s*:\s*(\d{6,12})', page.text)
        ids += re.findall(r'vimeo\.com/(\d{6,12})', page.text)
        seen_ids, unique_ids = set(), []
        for vid_id in ids:
            if vid_id not in seen_ids:
                seen_ids.add(vid_id)
                unique_ids.append(vid_id)

        videos = [
            {
                "url":       f"https://vimeo.com/{vid_id}",
                "embed_url": f"https://player.vimeo.com/video/{vid_id}",
                "mime":      "video/vimeo",
                "type":      "vimeo",
            }
            for vid_id in unique_ids[:max_videos]
        ]
        if not videos:
            return "No videos found on that Vimeo page.", []
        return f"Found {len(videos)} Vimeo video(s). They are displayed in the chat.", videos

    EXT_MIME = {
        ".mp4":"video/mp4",".webm":"video/webm",".ogg":"video/ogg",
        ".ogv":"video/ogg",".mov":"video/quicktime",".avi":"video/x-msvideo",
        ".mkv":"video/x-matroska",".m4v":"video/mp4",".3gp":"video/3gpp",
    }
    fetch_headers = {"User-Agent": "Mozilla/5.0 (compatible; OllamaGate/1.0)"}
    try:
        page = requests.get(url, timeout=12, headers=fetch_headers, allow_redirects=True)
        page.raise_for_status()
    except Exception as e:
        return f"Could not fetch page: {e}", []

    extractor = _HTMLVideoExtractor(base_url=url)
    extractor.feed(page.text)

    seen, unique = set(), []
    for s in extractor.videos:
        s = extractor.absolute(s.strip())
        if s and s not in seen:
            seen.add(s)
            unique.append(s)

    videos = []
    for vid_url in unique[:max_videos]:
        ext          = "." + vid_url.lower().split("?")[0].rsplit(".", 1)[-1] if "." in vid_url else ""
        guessed_mime = EXT_MIME.get(ext, "video/mp4")
        try:
            h  = requests.head(vid_url, timeout=6, headers=fetch_headers, allow_redirects=True)
            ct = h.headers.get("Content-Type", "").split(";")[0].strip()
            if ct.startswith("video/"):
                guessed_mime = ct
            elif ct and not ct.startswith("video/") and ext not in EXT_MIME:
                continue
        except Exception:
            pass
        videos.append({"url": vid_url, "mime": guessed_mime, "type": "direct"})

    if not videos:
        return "No videos found on that page.", []
    return f"Found {len(videos)} video(s) on {url}. They are displayed in the chat.", videos


def execute_tool(name: str, args: dict) -> tuple[str, list]:
    """Returns (text_result, images_or_videos_list).
    List items are {url,mime,b64} for images or {url,mime} for videos."""
    if name == "web_search":
        return tool_web_search(args.get("query","")), []
    elif name == "web_fetch":
        return tool_web_fetch(args.get("url","")), []
    elif name == "calculator":
        return tool_calculator(args.get("expression","")), []
    elif name == "get_datetime":
        return tool_get_datetime(), []
    elif name == "save_memory":
        return tool_save_memory(args.get("content",""), args.get("tags","")), []
    elif name == "fetch_images":
        return tool_fetch_images(args.get("url",""), args.get("max_images", 6), args.get("extensions",""))
    elif name == "fetch_videos":
        return tool_fetch_videos(args.get("url",""), args.get("max_videos", 4))
    else:
        return f"Unknown tool: {name}", []

PREMADE_PERSONAS = [
    {
        "id": "default",
        "label": "Default Assistant",
        "prompt": "You are a highly capable, adaptive, and witty AI collaborator. Your goal is to provide insightful, grounded responses that balance empathy with candor. Prioritize clarity and brevity, avoiding fluff. When you don't know something, be direct about it rather than speculating."
    },
    {
        "id": "roleplay",
        "label": "Roleplay (DM/GM)",
        "prompt": "You are an expert Dungeon Master and world-builder. Your narration is atmospheric and reactive, focusing on 'show, don't tell.' Use sensory details (smell, temperature, ambient sound) to ground the player. Respect player agency—never decide their actions for them—and maintain the 'Yes, and...' rule of improv to keep the story evolving logically."
    },
    {
        "id": "coder",
        "label": "Expert Coder",
        "prompt": "You are a senior full-stack architect with a focus on clean code, scalability, and security. When providing solutions, prioritize the DRY (Don't Repeat Yourself) principle and modern best practices. Include brief comments for complex logic, suggest potential edge cases, and explain the 'why' behind your architectural choices."
    },
    {
        "id": "tutor",
        "label": "Patient Tutor",
        "prompt": "You are a dedicated educator specializing in scaffolding. Instead of dumping information, build on what the user already knows. Use vivid analogies to explain abstract concepts. After explaining a segment, pause to ask a diagnostic question to ensure the user has grasped the core logic before moving forward."
    },
    {
        "id": "scientist",
        "label": "Research Scientist",
        "prompt": "You are a rigorous research scientist. Your tone is objective, analytical, and precise. Distinguish clearly between empirical evidence, established theory, and emerging hypotheses. When discussing data, mention sample sizes or methodology constraints where relevant, and always acknowledge the limits of current scientific consensus."
    },
    {
        "id": "creative",
        "label": "Creative Writer",
        "prompt": "You are a literary wordsmith with a mastery of pacing, subtext, and voice. Avoid clichés and 'purple prose'—instead, seek the most evocative and unexpected way to describe a moment. Adapt your prose style seamlessly to the genre requested, whether it's the grit of noir or the lyricism of high fantasy."
    },
    {
        "id": "socratic",
        "label": "Socratic Philosopher",
        "prompt": "You are a practitioner of the Socratic method. Your goal is to help the user achieve 'aporia'—the realization of their own hidden assumptions. Never lecture; instead, ask a series of disciplined, lean questions that force the user to define their terms and defend the internal logic of their own arguments."
    },
    {
        "id": "therapist",
        "label": "Supportive Listener",
        "prompt": "You are a compassionate listener trained in Rogerian (person-centered) techniques. Use active listening, mirroring, and validation to create a safe space. Do not jump to 'fixing' the problem; instead, help the user explore their emotions. Note: You are an AI, not a clinician; if a crisis is detected, prioritize safety and professional resources."
    },
]

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_BYTES

_state = {
    "token": None, "expires_at": None,
    "connected_ips": {}, "strikes": {},
}
_state_lock = threading.Lock()

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS chats (
            id              TEXT PRIMARY KEY,
            title           TEXT NOT NULL,
            model           TEXT NOT NULL,
            system_prompt   TEXT DEFAULT '',
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id     TEXT NOT NULL,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            thinking    TEXT DEFAULT '',
            created_at  TEXT NOT NULL,
            FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS attachments (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id  INTEGER NOT NULL,
            filename    TEXT,
            mime_type   TEXT,
            data_b64    TEXT,
            FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS analytics (
            ip              TEXT PRIMARY KEY,
            user_agent      TEXT,
            first_seen      TEXT,
            last_seen       TEXT,
            prompts_sent    INTEGER DEFAULT 0,
            login_count     INTEGER DEFAULT 0,
            country         TEXT DEFAULT '',
            city            TEXT DEFAULT '',
            hostname        TEXT DEFAULT '',
            isp             TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS token_sessions (
            session_id  TEXT PRIMARY KEY,
            ip          TEXT,
            created_at  TEXT,
            expires_at  TEXT
        );
        CREATE TABLE IF NOT EXISTS memories (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            content     TEXT NOT NULL,
            tags        TEXT DEFAULT '',
            created_at  TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS bash_snippets (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id     TEXT DEFAULT '',
            content     TEXT NOT NULL,
            label       TEXT DEFAULT '',
            created_at  TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS project_files (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT NOT NULL,
            language    TEXT DEFAULT '',
            content     TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );
        """)
        for col, defn in [
            ("system_prompt",   "TEXT DEFAULT ''"),
            ("thinking",        "TEXT DEFAULT ''"),
            ("login_count",     "INTEGER DEFAULT 0"),
            ("country",         "TEXT DEFAULT ''"),
            ("city",            "TEXT DEFAULT ''"),
            ("hostname",        "TEXT DEFAULT ''"),
            ("isp",             "TEXT DEFAULT ''"),
        ]:
            table = "chats" if col == "system_prompt" else \
                    "messages" if col == "thinking" else "analytics"
            _safe_add_column(db, table, col, defn)


def _safe_add_column(db, table, column, definition):
    try:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    except sqlite3.OperationalError:
        pass

def generate_token():
    letters   = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    digits    = "0123456789"
    specials  = "!@#$%^&*()_+}{][\":';?><,."
    alphabet  = letters + digits + specials
    mandatory = [
        secrets.choice(letters),
        secrets.choice(digits),
        secrets.choice(specials),
    ]
    rest = [secrets.choice(alphabet) for _ in range(TOKEN_LENGTH - len(mandatory))]
    token_chars = mandatory + rest
    for i in range(len(token_chars) - 1, 0, -1):
        j = secrets.randbelow(i + 1)
        token_chars[i], token_chars[j] = token_chars[j], token_chars[i]
    return "".join(token_chars)


def parse_duration(raw):
    raw = raw.strip().lower().replace(" ", "")
    m = re.fullmatch(r"(\d+)([hm])", raw)
    if not m:
        raise ValueError("Invalid format. Use e.g. '1 h' or '30 m'.")
    v, u = int(m.group(1)), m.group(2)
    return v * 3600 if u == "h" else v * 60


def _now():
    return datetime.now(timezone.utc)


def token_valid():
    return (_state["token"] and _state["expires_at"] and
            _now() < _state["expires_at"])


def seconds_remaining():
    if not token_valid(): return 0
    return max(0, int((_state["expires_at"] - _now()).total_seconds()))

def _fetch_geo_async(ip: str):
    """Resolve geo/hostname for an IP in background thread."""
    if ip in ("127.0.0.1", "::1", "localhost"):
        return
    def _worker():
        try:
            r = requests.get(
                f"http://ip-api.com/json/{ip}",
                params={"fields": "country,city,isp,reverse"},
                timeout=5
            )
            data = r.json()
            if data.get("status") == "success":
                with get_db() as db:
                    db.execute(
                        "UPDATE analytics SET country=?, city=?, isp=?, hostname=? WHERE ip=?",
                        (
                            data.get("country", ""),
                            data.get("city", ""),
                            data.get("isp", ""),
                            data.get("reverse", ""),
                            ip,
                        )
                    )
        except Exception:
            pass
    threading.Thread(target=_worker, daemon=True).start()


def record_analytics(ip: str, ua: str):
    now = _now().isoformat()
    with get_db() as db:
        existing = db.execute("SELECT ip FROM analytics WHERE ip=?", (ip,)).fetchone()
        if existing:
            db.execute(
                "UPDATE analytics SET last_seen=?, user_agent=?, login_count=login_count+1 WHERE ip=?",
                (now, ua, ip)
            )
        else:
            db.execute(
                "INSERT INTO analytics (ip, user_agent, first_seen, last_seen, prompts_sent, login_count) "
                "VALUES (?,?,?,?,0,1)",
                (ip, ua, now, now)
            )
    _fetch_geo_async(ip)


def increment_prompts(ip: str):
    with get_db() as db:
        db.execute(
            "UPDATE analytics SET prompts_sent=prompts_sent+1, last_seen=? WHERE ip=?",
            (_now().isoformat(), ip)
        )

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            return jsonify({"error": "Unauthorized"}), 401
        if not token_valid():
            session.clear()
            return jsonify({"error": "Session expired"}), 401
        return f(*args, **kwargs)
    return decorated


def get_client_ip():
    return request.headers.get("X-Forwarded-For", request.remote_addr).split(",")[0].strip()

def ollama_get_models():
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        models = []
        for m in r.json().get("models", []):
            name    = m["name"]
            details = m.get("details", {})
            size_gb = m.get("size", 0) / 1e9
            models.append({
                "name":           name,
                "family":         details.get("family", ""),
                "parameter_size": details.get("parameter_size", "?"),
                "quantization":   details.get("quantization_level", "?"),
                "size_gb":        round(size_gb, 1),
                "cloud":          model_is_cloud(name),
                "vision":         model_supports_vision(name),
                "thinking":       model_supports_thinking(name),
            })
        return models
    except Exception as e:
        print(f"[WARN] Ollama unreachable: {e}")
        return []


def model_supports_vision(name):
    return any(p in name.lower() for p in VISION_MODEL_PATTERNS)

def model_supports_thinking(name):
    return any(p in name.lower() for p in THINKING_MODEL_PATTERNS)

def model_is_cloud(name):
    return any(p in name.lower() for p in CLOUD_MODEL_PATTERNS)

def get_model_speed_hint(name):
    n = name.lower()
    for tag in ["405b","70b","72b","65b","34b","32b"]:
        if tag in n: return "slow"
    for tag in ["13b","14b","8b","7b"]:
        if tag in n: return "medium"
    for tag in ["3b","2b","1b","0.5b","mini","small","tiny","nano","phi"]:
        if tag in n: return "fast"
    return "medium"

def _extract_block(text, tag):
    pattern = rf"{re.escape(tag)}:\s*\{{\n?(.*?)\n?\}}"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""

def _extract_field(text, field):
    m = re.search(rf"^{re.escape(field)}\s*:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else ""

def _parse_stats_block(stats_text):
    result = {}
    for line in stats_text.splitlines():
        line = line.strip()
        if not line: continue
        m = re.match(r"(.+?):\s+(.+)", line)
        if m:
            k = m.group(1).strip().lower().replace(" ", "_")
            result[k] = m.group(2).strip()
    return result

def parse_benchmark_file(filepath):
    text = Path(filepath).read_text(encoding="utf-8", errors="replace")
    section_pattern = re.compile(
        r"^(?P<n>[^\n\{\}]+?)\n"
        r"(?P<meta>(?:[^\{\n][^\n]*\n){1,8})"
        r"(?:Think:\s*\{(?P<think>.*?)\})?\s*"
        r"Answer:\s*\{(?P<answer>.*?)\}\s*"
        r"Stats:\s*\{(?P<stats>.*?)\}",
        re.DOTALL | re.IGNORECASE | re.MULTILINE,
    )
    entries = []
    for m in section_pattern.finditer(text):
        name    = m.group("n").strip()
        meta    = m.group("meta") or ""
        think   = (m.group("think") or "").strip()
        answer  = (m.group("answer") or "").strip()
        stats_d = _parse_stats_block(m.group("stats") or "")
        def field(f): return _extract_field(meta, f)
        def yn(f):    return field(f).lower() in ("yes","true","1")
        eval_rate = 0.0
        rr = re.search(r"([\d.]+)", stats_d.get("eval_rate",""))
        if rr: eval_rate = float(rr.group(1))
        total_sec = 0.0
        ts = re.search(r"([\d.]+)", stats_d.get("total_duration",""))
        if ts: total_sec = float(ts.group(1))
        entries.append({
            "model": name, "thinking": yn("thinking"), "vision": yn("vision"),
            "uncensored": field("uncensored"), "cloud": yn("cloud"),
            "parameters": field("parameter") or field("parameters"),
            "quantization": field("quantization"), "think_block": think,
            "answer": answer, "stats": stats_d, "eval_rate": eval_rate,
            "total_sec": total_sec, "prompt": field("prompt"),
        })
    return entries

def load_all_benchmarks():
    results = []
    for f in sorted(Path(BENCHMARK_DIR).glob("*.txt")):
        try:
            entries = parse_benchmark_file(f)
            for e in entries: e["source_file"] = f.name
            results.extend(entries)
        except Exception as ex:
            print(f"[WARN] Could not parse {f}: {ex}")
    return results

@app.route("/")
def index():
    if session.get("authenticated") and token_valid():
        return render_template("chat.html")
    return render_template("login.html")


@app.route("/stats")
def stats_page():
    if not (session.get("authenticated") and token_valid()):
        return render_template("login.html")
    return render_template("stats.html")


@app.route("/api/login", methods=["POST"])
def api_login():
    ip = get_client_ip()
    with _state_lock:
        if _state["strikes"].get(ip, 0) >= MAX_LOGIN_STRIKES:
            return jsonify({"error": "Too many failed attempts. Access blocked."}), 429

    data      = request.get_json(silent=True) or {}
    submitted = data.get("token", "")

    def fail():
        with _state_lock:
            _state["strikes"][ip] = _state["strikes"].get(ip, 0) + 1
        return jsonify({"error": "Wrong token."}), 401

    if not token_valid():
        return jsonify({"error": "No active session on server."}), 401
    if not secrets.compare_digest(submitted, _state["token"]):
        return fail()

    with _state_lock:
        _state["strikes"].pop(ip, None)
        _state["connected_ips"][ip] = {
            "connected_at": _now().isoformat(),
            "requests": 0,
            "ua": request.headers.get("User-Agent", "")
        }

    session["authenticated"] = True
    session["ip"] = ip
    record_analytics(ip, request.headers.get("User-Agent", ""))
    return jsonify({"ok": True})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    ip = session.get("ip")
    if ip:
        with _state_lock: _state["connected_ips"].pop(ip, None)
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    return jsonify({
        "token_valid":     token_valid(),
        "expires_in":      seconds_remaining(),
        "connected_count": len(_state["connected_ips"]),
    })

@app.route("/api/models")
@login_required
def api_models():
    local  = ollama_get_models()
    clouds = [
        {"name": k, "cloud": True, "vision": True,
         "thinking": "o1" in k or "o3" in k,
         "parameter_size": v["params"], "quantization": "cloud",
         "size_gb": 0, "family": v["provider"], "speed_hint": v["speed"]}
        for k, v in CLOUD_MODELS.items()
    ]
    all_models = local + clouds
    for m in all_models:
        if "speed_hint" not in m:
            m["speed_hint"] = get_model_speed_hint(m["name"])
    return jsonify({
        "models":        all_models,
        "local_count":   len(local),
        "cloud_count":   len(clouds),
        "vision_models": [m["name"] for m in all_models if m.get("vision")],
    })


@app.route("/api/model/vision/<path:model_name>")
@login_required
def api_model_vision(model_name):
    return jsonify({"supports_vision": model_supports_vision(model_name)})


@app.route("/api/ps")
@login_required
def api_ps():
    """Proxy Ollama's /api/ps so the frontend can see what's currently loaded."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/ps", timeout=5)
        return jsonify(r.json() if r.ok else {"models": []})
    except Exception as e:
        return jsonify({"models": [], "error": str(e)})


@app.route("/api/model/unload", methods=["POST"])
@login_required
def api_model_unload():
    """Explicitly unload one or all models from Ollama VRAM.

    Body (optional):
      { "model": "llama3:8b" }   → unload only that model
      {}                         → unload all loaded models
    """
    data  = request.get_json(silent=True) or {}
    name  = data.get("model", "").strip()

    if name:
        ok = _unload_model(name, timeout=120)
        return jsonify({"ok": ok, "unloaded": [name] if ok else []})
    else:
        running = _ollama_running_models()
        _unload_ollama_models(timeout=120)
        return jsonify({"ok": True, "unloaded": running})

@app.route("/api/personas")
@login_required
def api_personas():
    return jsonify({"personas": PREMADE_PERSONAS})


@app.route("/api/benchmarks")
@login_required
def api_benchmarks():
    data = load_all_benchmarks()
    return jsonify({"benchmarks": data, "count": len(data)})


@app.route("/api/benchmarks/upload", methods=["POST"])
@login_required
def api_benchmarks_upload():
    f = request.files.get("file")
    if not f: return jsonify({"error": "No file"}), 400
    bd = Path(BENCHMARK_DIR); bd.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\-.]", "_", f.filename or "upload.txt")
    dest = bd / safe; f.save(dest)
    try:
        entries = parse_benchmark_file(dest)
        return jsonify({"ok": True, "entries": len(entries), "filename": safe})
    except Exception as e:
        return jsonify({"error": str(e)}), 422

@app.route("/api/memory", methods=["GET"])
@login_required
def api_list_memories():
    with get_db() as db:
        rows = db.execute(
            "SELECT id, content, tags, created_at FROM memories ORDER BY id DESC"
        ).fetchall()
    return jsonify({"memories": [dict(r) for r in rows]})


@app.route("/api/memory", methods=["POST"])
@login_required
def api_add_memory():
    data    = request.get_json(silent=True) or {}
    content = data.get("content", "").strip()
    tags    = data.get("tags", "").strip()
    if not content:
        return jsonify({"error": "content required"}), 400
    with get_db() as db:
        cur = db.execute(
            "INSERT INTO memories (content, tags, created_at) VALUES (?,?,?)",
            (content, tags, _now().isoformat())
        )
    return jsonify({"ok": True, "id": cur.lastrowid})


@app.route("/api/memory/<int:mid>", methods=["DELETE"])
@login_required
def api_delete_memory(mid):
    with get_db() as db:
        db.execute("DELETE FROM memories WHERE id=?", (mid,))
    return jsonify({"ok": True})


@app.route("/api/memory", methods=["DELETE"])
@login_required
def api_clear_memories():
    with get_db() as db:
        db.execute("DELETE FROM memories")
    return jsonify({"ok": True})

@app.route("/api/bash_snippets", methods=["GET"])
@login_required
def api_list_bash_snippets():
    with get_db() as db:
        rows = db.execute(
            "SELECT id, chat_id, content, label, created_at FROM bash_snippets ORDER BY id DESC"
        ).fetchall()
    return jsonify({"snippets": [dict(r) for r in rows]})


@app.route("/api/bash_snippets", methods=["POST"])
@login_required
def api_add_bash_snippet():
    data    = request.get_json(silent=True) or {}
    content = data.get("content", "").strip()
    label   = data.get("label", "").strip()
    chat_id = data.get("chat_id", "")
    if not content:
        return jsonify({"error": "content required"}), 400
    with get_db() as db:
        cur = db.execute(
            "INSERT INTO bash_snippets (chat_id, content, label, created_at) VALUES (?,?,?,?)",
            (chat_id, content, label, _now().isoformat())
        )
    return jsonify({"ok": True, "id": cur.lastrowid})


@app.route("/api/bash_snippets/<int:sid>", methods=["PUT"])
@login_required
def api_update_bash_snippet(sid):
    data    = request.get_json(silent=True) or {}
    content = data.get("content", "").strip()
    label   = data.get("label", "").strip()
    if not content:
        return jsonify({"error": "content required"}), 400
    with get_db() as db:
        db.execute(
            "UPDATE bash_snippets SET content=?, label=? WHERE id=?",
            (content, label, sid)
        )
    return jsonify({"ok": True})


@app.route("/api/bash_snippets/<int:sid>", methods=["DELETE"])
@login_required
def api_delete_bash_snippet(sid):
    with get_db() as db:
        db.execute("DELETE FROM bash_snippets WHERE id=?", (sid,))
    return jsonify({"ok": True})


@app.route("/api/bash_snippets", methods=["DELETE"])
@login_required
def api_clear_bash_snippets():
    with get_db() as db:
        db.execute("DELETE FROM bash_snippets")
    return jsonify({"ok": True})

@app.route("/api/files", methods=["GET"])
@login_required
def api_list_files():
    with get_db() as db:
        rows = db.execute(
            "SELECT id, filename, language, content, created_at, updated_at FROM project_files ORDER BY updated_at DESC"
        ).fetchall()
    return jsonify({"files": [dict(r) for r in rows]})


@app.route("/api/files", methods=["POST"])
@login_required
def api_create_file():
    data     = request.get_json(silent=True) or {}
    filename = data.get("filename", "").strip()
    language = data.get("language", "").strip()
    content  = data.get("content", "")
    if not filename:
        return jsonify({"error": "filename required"}), 400
    now = _now().isoformat()
    with get_db() as db:
        existing = db.execute("SELECT id FROM project_files WHERE filename=?", (filename,)).fetchone()
        if existing:
            db.execute(
                "UPDATE project_files SET language=?, content=?, updated_at=? WHERE id=?",
                (language, content, now, existing["id"])
            )
            fid = existing["id"]
        else:
            cur = db.execute(
                "INSERT INTO project_files (filename, language, content, created_at, updated_at) VALUES (?,?,?,?,?)",
                (filename, language, content, now, now)
            )
            fid = cur.lastrowid
    return jsonify({"ok": True, "id": fid})


@app.route("/api/files/<int:fid>", methods=["PUT"])
@login_required
def api_update_file(fid):
    data    = request.get_json(silent=True) or {}
    content = data.get("content", "")
    lang    = data.get("language", None)
    now     = _now().isoformat()
    with get_db() as db:
        if lang is not None:
            db.execute("UPDATE project_files SET content=?, language=?, updated_at=? WHERE id=?",
                       (content, lang, now, fid))
        else:
            db.execute("UPDATE project_files SET content=?, updated_at=? WHERE id=?",
                       (content, now, fid))
    return jsonify({"ok": True})


@app.route("/api/files/<int:fid>", methods=["DELETE"])
@login_required
def api_delete_file(fid):
    with get_db() as db:
        db.execute("DELETE FROM project_files WHERE id=?", (fid,))
    return jsonify({"ok": True})


@app.route("/api/files", methods=["DELETE"])
@login_required
def api_clear_files():
    with get_db() as db:
        db.execute("DELETE FROM project_files")
    return jsonify({"ok": True})

@app.route("/api/chats", methods=["GET"])
@login_required
def api_list_chats():
    with get_db() as db:
        rows = db.execute(
            "SELECT id,title,model,system_prompt,created_at,updated_at "
            "FROM chats ORDER BY updated_at DESC"
        ).fetchall()
    return jsonify({"chats": [dict(r) for r in rows]})


@app.route("/api/chats", methods=["POST"])
@login_required
def api_create_chat():
    data   = request.get_json(silent=True) or {}
    model  = data.get("model","")
    title  = data.get("title","New Chat")
    system = data.get("system_prompt","")
    if not model: return jsonify({"error":"model required"}), 400
    cid = str(uuid.uuid4()); now = _now().isoformat()
    with get_db() as db:
        db.execute(
            "INSERT INTO chats (id,title,model,system_prompt,created_at,updated_at) VALUES (?,?,?,?,?,?)",
            (cid, title, model, system, now, now)
        )
    return jsonify({"id":cid,"title":title,"model":model,"system_prompt":system})


@app.route("/api/chats/<cid>", methods=["GET"])
@login_required
def api_get_chat(cid):
    with get_db() as db:
        chat = db.execute("SELECT * FROM chats WHERE id=?", (cid,)).fetchone()
        if not chat: return jsonify({"error":"Not found"}), 404
        msgs = db.execute(
            "SELECT m.id,m.role,m.content,m.thinking,m.created_at,"
            "       a.filename,a.mime_type,a.data_b64 "
            "FROM messages m LEFT JOIN attachments a ON a.message_id=m.id "
            "WHERE m.chat_id=? ORDER BY m.id", (cid,)
        ).fetchall()
    return jsonify({"chat":dict(chat),"messages":[dict(m) for m in msgs]})


@app.route("/api/chats/<cid>", methods=["DELETE"])
@login_required
def api_delete_chat(cid):
    with get_db() as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("DELETE FROM attachments WHERE message_id IN (SELECT id FROM messages WHERE chat_id=?)", (cid,))
        db.execute("DELETE FROM messages WHERE chat_id=?", (cid,))
        db.execute("DELETE FROM chats WHERE id=?", (cid,))
    return jsonify({"ok":True})


@app.route("/api/chats/<cid>/clean", methods=["POST"])
@login_required
def api_clean_chat(cid):
    """Delete all messages in a chat (keeps the chat itself)."""
    with get_db() as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("DELETE FROM attachments WHERE message_id IN (SELECT id FROM messages WHERE chat_id=?)", (cid,))
        db.execute("DELETE FROM messages WHERE chat_id=?", (cid,))
        db.execute("UPDATE chats SET updated_at=? WHERE id=?", (_now().isoformat(), cid))
    return jsonify({"ok": True, "message": "Chat history cleared."})


@app.route("/api/chats/<cid>/title", methods=["PATCH"])
@login_required
def api_rename_chat(cid):
    data  = request.get_json(silent=True) or {}
    title = data.get("title","").strip()
    if not title: return jsonify({"error":"title required"}), 400
    with get_db() as db:
        db.execute("UPDATE chats SET title=? WHERE id=?", (title, cid))
    return jsonify({"ok":True})


@app.route("/api/chats/<cid>/system", methods=["PATCH"])
@login_required
def api_update_system(cid):
    data   = request.get_json(silent=True) or {}
    system = data.get("system_prompt","")
    with get_db() as db:
        db.execute("UPDATE chats SET system_prompt=? WHERE id=?", (system, cid))
    return jsonify({"ok":True})

@app.route("/api/chats/<cid>/message", methods=["POST"])
@login_required
def api_send_message(cid):
    ip = get_client_ip()

    if request.content_type and "multipart/form-data" in request.content_type:
        user_content      = request.form.get("content","").strip()
        model             = request.form.get("model","").strip()
        extended_thinking = request.form.get("extended_thinking","false").lower() == "true"
        use_tools         = request.form.get("use_tools","false").lower() == "true"
        use_memory        = request.form.get("use_memory","false").lower() == "true"
        file_obj          = request.files.get("file")
        num_ctx    = request.form.get("num_ctx", 8192, type=int)
    else:
        data              = request.get_json(silent=True) or {}
        user_content      = data.get("content","").strip()
        model             = data.get("model","").strip()
        extended_thinking = bool(data.get("extended_thinking", False))
        use_tools         = bool(data.get("use_tools", False))
        use_memory        = bool(data.get("use_memory", False))
        file_obj          = None
        num_ctx    = data.get("num_ctx", 8192)

    if not user_content and not file_obj:
        return jsonify({"error":"content required"}), 400

    attach_filename = attach_mime = attach_b64 = image_b64 = None
    if file_obj:
        fname    = file_obj.filename or "file"
        mime     = file_obj.mimetype or mimetypes.guess_type(fname)[0] or "application/octet-stream"
        file_obj.seek(0,2); size = file_obj.tell(); file_obj.seek(0)
        if size > MAX_CONTENT_BYTES:
            if mime.startswith("video/") and size <= MAX_VIDEO_BYTES:
                pass
            else:
                return jsonify({"error":f"File exceeds {MAX_CONTENT_MB}MB limit."}), 413
        if not any(mime.startswith(p) for p in ALLOWED_MIME_PREFIXES):
            return jsonify({"error":f"File type '{mime}' not allowed."}), 415
        raw_bytes       = file_obj.read()
        attach_b64      = base64.b64encode(raw_bytes).decode()
        attach_filename = fname
        attach_mime     = mime
        if mime.startswith("image/"):
            if not model_supports_vision(model):
                return jsonify({"error":"This model does not support images."}), 422
            image_b64 = attach_b64
        elif mime.startswith("video/"):
            user_content += f"\n\n[Video attached: {fname} ({mime})]"
        elif mime == "application/pdf":
            user_content += "\n\n[PDF attached: " + fname + "]"
        else:
            text_content  = raw_bytes.decode("utf-8", errors="replace")
            user_content += f"\n\n--- File: {fname} ---\n{text_content[:8000]}"

    with get_db() as db:
        chat = db.execute("SELECT * FROM chats WHERE id=?", (cid,)).fetchone()
        if not chat: return jsonify({"error":"Chat not found"}), 404
        hist = db.execute(
            "SELECT role,content FROM messages WHERE chat_id=? ORDER BY id", (cid,)
        ).fetchall()

    system_prompt = chat["system_prompt"] or ""

    if use_memory:
        with get_db() as db:
            mems = db.execute(
                "SELECT content, tags FROM memories ORDER BY id ASC LIMIT 60"
            ).fetchall()
        if mems:
            seen = set()
            unique_mems = []
            for m in mems:
                key = m['content'].strip().lower()
                if key not in seen:
                    seen.add(key)
                    unique_mems.append(m)
            mem_lines_parts = []
            total_chars = 0
            for m in unique_mems:
                line = f"- {m['content']}" + (f" [{m['tags']}]" if m['tags'] else "")
                if total_chars + len(line) > 2000:
                    mem_lines_parts.append("- [additional memories trimmed to stay within context limit]")
                    break
                mem_lines_parts.append(line)
                total_chars += len(line)

            mem_lines = "\n".join(mem_lines_parts)
            memory_block = (
                "=== PERSISTENT MEMORY (reference only) ===\n"
                "These facts were saved from prior conversations. "
                "Use them silently to inform your answers — do NOT recite, list, or repeat them unless directly asked.\n"
                f"{mem_lines}\n"
                "=== END MEMORY ===\n\n"
            )
            system_prompt = memory_block + system_prompt

    with get_db() as db:
        pfiles = db.execute(
            "SELECT filename, language, content FROM project_files ORDER BY updated_at DESC LIMIT 15"
        ).fetchall()
    if pfiles:
        file_block = "\n\n=== PROJECT FILES (current state — use these as the authoritative source when editing) ===\n"
        for pf in pfiles:
            lang_tag = pf["language"] or ""
            file_block += f"\n--- {pf['filename']} ---\n```{lang_tag}\n{pf['content']}\n```\n"
        file_block += "=== END PROJECT FILES ===\n"
        system_prompt = (system_prompt + file_block).strip() if system_prompt else file_block.strip()

    if extended_thinking:
        think_prefix = (
            "Before answering, reason through this problem step by step inside <think>...</think> tags. "
            "Explore multiple approaches, consider edge cases, and verify your reasoning. "
            "After your thinking, provide a clear and thorough answer.\n\n"
        )
        system_prompt = think_prefix + system_prompt

    ollama_messages = []
    if system_prompt:
        ollama_messages.append({"role": "system", "content": system_prompt})
    ollama_messages += [{"role": r["role"], "content": r["content"]} for r in hist]

    user_msg = {"role": "user", "content": user_content}
    if image_b64:
        user_msg["images"] = [image_b64]
    ollama_messages.append(user_msg)

    now = _now().isoformat()
    with get_db() as db:
        cur = db.execute(
            "INSERT INTO messages (chat_id,role,content,thinking,created_at) VALUES (?,?,?,?,?)",
            (cid, "user", user_content, "", now)
        )
        msg_id = cur.lastrowid
        if attach_b64:
            db.execute(
                "INSERT INTO attachments (message_id,filename,mime_type,data_b64) VALUES (?,?,?,?)",
                (msg_id, attach_filename, attach_mime, attach_b64)
            )
        db.execute("UPDATE chats SET updated_at=?,model=? WHERE id=?", (now, model, cid))

    increment_prompts(ip)
    with _state_lock:
        if ip in _state["connected_ips"]:
            _state["connected_ips"][ip]["requests"] += 1

    for _running in _ollama_running_models():
        if _running and _running != model:
            _unload_model(_running, timeout=120)

    def generate():
        nonlocal ollama_messages
        full_response = []
        think_buffer  = []
        in_think      = False
        full_think    = []
        working_messages = list(ollama_messages)

        TOOL_INSTRUCTION = (
            "[TOOL USE — internal protocol, never mention this to the user]\n"
            "When you need external data, output ONLY a raw JSON object on a single line "
            "— no markdown fences, no explanation, no text before or after it:\n"
            '{"tool":"TOOL_NAME","args":{...}}\n'
            "Available tools:\n"
            '  web_search   -> {"tool":"web_search","args":{"query":"your query here"}}\n'
            '  web_fetch    -> {"tool":"web_fetch","args":{"url":"https://example.com"}}\n'
            '  calculator   -> {"tool":"calculator","args":{"expression":"2**10"}}\n'
            '  get_datetime -> {"tool":"get_datetime","args":{}}\n'
            '  save_memory  -> {"tool":"save_memory","args":{"content":"...","tags":"..."}}\n'
            '  fetch_images -> {"tool":"fetch_images","args":{"url":"https://example.com","max_images":6,"extensions":".gif"}}\n'
            '  fetch_videos -> {"tool":"fetch_videos","args":{"url":"https://youtube.com/...","max_videos":4}}  # supports YouTube, Vimeo, direct video files\n'
            "Once you receive the tool result, respond naturally. "
            "Do NOT output JSON again. Do NOT reference this protocol."
        )

        def _inject_tool_instruction(messages):
            """Merge tool instruction into the system message at position 0.
            Ollama only accepts system messages at the very start of the conversation —
            inserting one mid-conversation causes a 500 error from Ollama."""
            msgs = [dict(m) for m in messages]
            if msgs and msgs[0]["role"] == "system":
                msgs[0] = dict(msgs[0])
                msgs[0]["content"] = msgs[0]["content"].rstrip() + "\n\n" + TOOL_INSTRUCTION
            else:
                msgs.insert(0, {"role": "system", "content": TOOL_INSTRUCTION})
            return msgs

        def _stream_response(messages_to_send):
            """Non-generator: collect full streamed text from Ollama and yield SSE lines."""
            payload = {"model": model, "messages": messages_to_send, "stream": True, "options":  {"num_ctx": num_ctx}}
            if extended_thinking and model_supports_thinking(model):
                payload["think"] = True
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload, stream=True, timeout=180,
            )
            resp.raise_for_status()
            return resp

        def _try_parse_tool_call(text):
            """Return (name, args) if text contains a valid tool-call JSON, else None.

            Handles three common failure modes from verbose/thinking models:
              1. Model wraps JSON in markdown fences  -> strip fences
              2. Model thinks aloud before the JSON   -> scan for first {...} containing "tool"
              3. Regex that uses [^{}]* misses nested args:{...} -> walk braces properly
            """
            t = text.strip()
            if t.startswith("```"):
                t = re.sub(r"^```[^\n]*\n?", "", t).rstrip("`").strip()

            try:
                obj = json.loads(t)
                if isinstance(obj, dict) and "tool" in obj:
                    return obj["tool"], _clean_args(obj.get("args", {}))
            except Exception:
                pass
            
            i = 0
            while i < len(t):
                if t[i] != '{':
                    i += 1
                    continue
                depth, j = 0, i
                while j < len(t):
                    if t[j] == '{':
                        depth += 1
                    elif t[j] == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = t[i:j + 1]
                            try:
                                obj = json.loads(candidate)
                                if isinstance(obj, dict) and "tool" in obj:
                                    return obj["tool"], _clean_args(obj.get("args", {}))
                            except Exception:
                                pass
                            break
                    j += 1
                i += 1
            return None

        def _clean_args(args: dict) -> dict:
            """Strip markdown decoration artefacts that models sometimes add to arg values
            (e.g. __https://url__ -> https://url)."""
            cleaned = {}
            for k, v in args.items():
                if isinstance(v, str):
                    v = v.strip().strip("_").strip()
                cleaned[k] = v
            return cleaned

        def _handle_tool_media(name, args, result_text, result_media, working_msgs, use_native):
            """Append tool result to working_msgs and yield SSE events for media.
            use_native=True  -> role:'tool' messages (Ollama native format)
            use_native=False -> role:'user' messages (fallback JSON format)
            """
            events = []
            if result_media and name == "fetch_images":
                events.append(json.dumps({"tool_images": result_media}))
                if use_native:
                    if model_supports_vision(model):
                        vision_images = [m["b64"] for m in result_media if m.get("b64")][:2]
                        if vision_images:
                            working_msgs.append({
                                "role": "tool",
                                "content": f"{result_text[:500]}\n(Images attached below.)",
                            })
                        else:
                            working_msgs.append({"role": "tool", "content": result_text[:1000]})
                    else:
                        working_msgs.append({
                            "role": "tool",
                            "content": f"{result_text[:1000]}\n(Images were shown to the user; you cannot see them.)"
                        })
                else:
                    if model_supports_vision(model):
                        vision_images = [m["b64"] for m in result_media if m.get("b64")][:2]
                        if vision_images:
                            working_msgs.append({
                                "role": "user",
                                "content": f"[Tool result for {name}]:\n{result_text[:500]}\nImages attached.",
                                "images": vision_images,
                            })
                        else:
                            working_msgs.append({"role": "user", "content": f"[Tool result for {name}]:\n{result_text[:1000]}"})
                    else:
                        working_msgs.append({
                            "role": "user",
                            "content": f"[Tool result for {name}]:\n{result_text[:1000]}\n(Images shown to user; you cannot see them.)"
                        })

            elif result_media and name == "fetch_videos":
                events.append(json.dumps({"tool_videos": result_media}))
                if use_native:
                    working_msgs.append({"role": "tool", "content": result_text[:1000]})
                else:
                    working_msgs.append({"role": "user", "content": f"[Tool result for {name}]:\n{result_text[:1000]}"})

            else:
                if use_native:
                    working_msgs.append({"role": "tool", "content": result_text[:8000]})
                else:
                    working_msgs.append({
                        "role": "user",
                        "content": (
                            f"[Tool result for {name}]:\n{result_text[:8000]}\n\n"
                            "Now answer the original question using this information. "
                            "Do NOT output JSON. Respond naturally."
                        )
                    })
            return events

        try:
            if use_tools:
                _native_tools_ok = True

                for _iteration in range(MAX_TOOL_ITERATIONS):
                    if _iteration == 0:
                        yield f"data: {json.dumps({'thinking_delta': 'Analyzing request...\n'})}\n\n"
                    elif _iteration == MAX_TOOL_ITERATIONS - 1:
                        yield f"data: {json.dumps({'thinking_delta': 'Finalizing answer...\n'})}\n\n"
                    else:
                        yield f"data: {json.dumps({'thinking_delta': f'Tool call {_iteration} complete — continuing...\n'})}\n\n"

                    payload = {
                        "model":    model,
                        "messages": working_messages if _native_tools_ok else _inject_tool_instruction(working_messages),
                        "stream":   False,
                        "options":  {"num_ctx": num_ctx},
                    }
                    if extended_thinking and model_supports_thinking(model):
                        payload["think"] = True
                    if _native_tools_ok and _iteration < MAX_TOOL_ITERATIONS - 1:
                        payload["tools"] = TOOLS_DEFS

                    resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120)

                    if resp.status_code == 500 and _native_tools_ok and "tools" in payload:
                        _native_tools_ok = False
                        payload.pop("tools")
                        payload["messages"] = _inject_tool_instruction(working_messages)
                        resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120)

                    if resp.status_code != 200:
                        yield f"data: {json.dumps({'error': f'Ollama Error {resp.status_code}: request failed.'})}\n\n"
                        break

                    resp_json    = resp.json()
                    msg_data     = resp_json.get("message", {})
                    msg_text     = msg_data.get("content", "").strip()
                    native_think = msg_data.get("thinking", "")

                    if native_think:
                        full_think.append(native_think)
                        yield f"data: {json.dumps({'thinking_delta': native_think + '\n'})}\n\n"

                    think_match = re.search(r"<think>(.*?)(?:</think>|$)", msg_text, re.DOTALL)
                    if think_match:
                        think_content = think_match.group(1).strip()
                        if think_content:
                            full_think.append(think_content)
                            yield f"data: {json.dumps({'thinking_delta': think_content + '\n'})}\n\n"

                    native_tool_calls = msg_data.get("tool_calls") or []

                    if native_tool_calls:
                        working_messages.append({
                            "role":       "assistant",
                            "content":    msg_text,
                            "tool_calls": native_tool_calls,
                        })

                        for tc in native_tool_calls:
                            fn   = tc.get("function", {})
                            name = fn.get("name", "")
                            args = fn.get("arguments", {})
                            if isinstance(args, str):
                                try: args = json.loads(args)
                                except: args = {}

                            yield f"data: {json.dumps({'tool_call': {'name': name, 'args': args}})}\n\n"
                            result_text, result_media = execute_tool(name, args)

                            media_events = _handle_tool_media(
                                name, args, result_text, result_media,
                                working_messages, use_native=True
                            )
                            for ev in media_events:
                                yield f"data: {ev}\n\n"

                            yield f"data: {json.dumps({'tool_result': {'name': name, 'result': result_text[:500]}})}\n\n"
                        continue

                    tool_call = _try_parse_tool_call(msg_text)

                    if tool_call is not None:
                        name, args = tool_call
                        if isinstance(args, str):
                            try: args = json.loads(args)
                            except: args = {}

                        yield f"data: {json.dumps({'tool_call': {'name': name, 'args': args}})}\n\n"
                        result_text, result_media = execute_tool(name, args)

                        working_messages.append({"role": "assistant", "content": msg_text})
                        media_events = _handle_tool_media(
                            name, args, result_text, result_media,
                            working_messages, use_native=False
                        )
                        for ev in media_events:
                            yield f"data: {ev}\n\n"

                        yield f"data: {json.dumps({'tool_result': {'name': name, 'result': result_text[:500]}})}\n\n"
                        continue

                    answer = re.sub(r"<think>.*?</think>", "", msg_text, flags=re.DOTALL).strip()
                    answer = re.sub(r"<think>.*",          "", answer,   flags=re.DOTALL).strip()

                    if answer:
                        full_response.append(answer)
                        yield f"data: {json.dumps({'delta': answer})}\n\n"
                    break

            else:
                payload = {
                    "model":    model,
                    "messages": working_messages,
                    "stream":   True,
                    "options":  {"num_ctx": num_ctx}
                }
                if extended_thinking and model_supports_thinking(model):
                    payload["think"] = True

                resp = requests.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json=payload, stream=True, timeout=180,
                )
                resp.raise_for_status()

                tag_buf       = ""
                think_started = False
                in_think      = False

                _lq = queue.Queue()
                def _ollama_reader():
                    try:
                        for _l in resp.iter_lines():
                            _lq.put(_l)
                    except Exception as _exc:
                        _lq.put(_exc)
                    finally:
                        _lq.put(None)
                threading.Thread(target=_ollama_reader, daemon=True).start()

                while True:
                    try:
                        _item = _lq.get(timeout=15)
                    except queue.Empty:
                        yield ": keep-alive\n\n"
                        continue
                    if _item is None:
                        break
                    if isinstance(_item, BaseException):
                        raise _item
                    line = _item
                    if not line: continue
                    try: chunk = json.loads(line)
                    except: continue

                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        tag_buf += delta

                        while tag_buf:
                            if not think_started:
                                idx = tag_buf.find("<think>")
                                if idx >= 0:
                                    before = tag_buf[:idx]
                                    if before:
                                        full_response.append(before)
                                        yield f"data: {json.dumps({'delta': before})}\n\n"
                                    tag_buf       = tag_buf[idx + 7:]
                                    think_started = True
                                    in_think      = True
                                    yield f"data: {json.dumps({'thinking_start': True})}\n\n"
                                else:
                                    safe = max(0, len(tag_buf) - 6)
                                    if safe:
                                        out = tag_buf[:safe]
                                        full_response.append(out)
                                        yield f"data: {json.dumps({'delta': out})}\n\n"
                                        tag_buf = tag_buf[safe:]
                                    break

                            elif in_think:
                                idx = tag_buf.find("</think>")
                                if idx >= 0:
                                    think_part = tag_buf[:idx]
                                    if think_part:
                                        full_think.append(think_part)
                                        yield f"data: {json.dumps({'thinking_delta': think_part})}\n\n"
                                    tag_buf  = tag_buf[idx + 8:]
                                    in_think = False
                                    yield f"data: {json.dumps({'thinking_end': True})}\n\n"
                                else:
                                    safe = max(0, len(tag_buf) - 8)
                                    if safe:
                                        out = tag_buf[:safe]
                                        full_think.append(out)
                                        yield f"data: {json.dumps({'thinking_delta': out})}\n\n"
                                        tag_buf = tag_buf[safe:]
                                    break

                            else:
                                full_response.append(tag_buf)
                                yield f"data: {json.dumps({'delta': tag_buf})}\n\n"
                                tag_buf = ""

                    native_think = chunk.get("message", {}).get("thinking", "")
                    if native_think:
                        if not think_started:
                            think_started = True
                            yield f"data: {json.dumps({'thinking_start': True})}\n\n"
                        full_think.append(native_think)
                        yield f"data: {json.dumps({'thinking_delta': native_think})}\n\n"

                    if chunk.get("done"):
                        if tag_buf:
                            if in_think:
                                full_think.append(tag_buf)
                                yield f"data: {json.dumps({'thinking_delta': tag_buf})}\n\n"
                            else:
                                full_response.append(tag_buf)
                                yield f"data: {json.dumps({'delta': tag_buf})}\n\n"
                        if think_started:
                            yield f"data: {json.dumps({'thinking_end': True})}\n\n"
                        stats = {k: chunk.get(k) for k in [
                            "total_duration", "load_duration", "prompt_eval_count",
                            "prompt_eval_duration", "eval_count", "eval_duration"
                        ]}
                        yield f"data: {json.dumps({'stats': stats})}\n\n"
                        break

        except Exception as e:
            yield f"data: {json.dumps({'error':str(e)})}\n\n"
            return

        assistant_text = "".join(full_response)
        thinking_text  = "".join(full_think)
        save_now = _now().isoformat()
        with get_db() as db:
            db.execute(
                "INSERT INTO messages (chat_id,role,content,thinking,created_at) VALUES (?,?,?,?,?)",
                (cid, "assistant", assistant_text, thinking_text, save_now)
            )
            row = db.execute("SELECT title FROM chats WHERE id=?", (cid,)).fetchone()
            if row and row["title"] == "New Chat" and user_content:
                db.execute("UPDATE chats SET title=? WHERE id=?",
                           (user_content[:50].replace("\n"," "), cid))

        yield f"data: {json.dumps({'done':True})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"},
    )

SAFE_RESOLUTIONS = {
    (824, 824), (640, 640),
    (824, 464), (768, 512),
    (464, 824), (512, 768),
}

IMAGEGEN_MODELS = {
    "anime": {
        "repo": "UnfilteredAI/NSFW-GEN-ANIME",
        "device_map": True,
        "variant": None,
        "safetensors": False,
    },
    "sdxl": {
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "device_map": False,
        "variant": "fp16",
        "safetensors": True,
    },
    "v2": {
        "repo": "UnfilteredAI/NSFW-gen-v2",
        "device_map": True,
        "variant": None,
        "safetensors": False,
    },
}

_img_lock   = threading.Lock()
_pipe_cache = {"key": None, "pipe": None}
_img_jobs   = {}


def _load_pipeline(model_key: str):
    """Load (or reuse) a diffusion pipeline. Must be called inside _img_lock."""
    if not _DIFFUSERS_OK:
        raise RuntimeError("diffusers / torch not installed.")
    cfg = IMAGEGEN_MODELS[model_key]
    if _pipe_cache["key"] == model_key and _pipe_cache["pipe"] is not None:
        return _pipe_cache["pipe"]

    if _pipe_cache["pipe"] is not None:
        del _pipe_cache["pipe"]
        _pipe_cache["pipe"] = None
        _pipe_cache["key"]  = None
        if _DIFFUSERS_OK:
            torch.cuda.empty_cache()

    kwargs = {"torch_dtype": torch.float16}
    if cfg["variant"]:
        kwargs["variant"] = cfg["variant"]
    if cfg["safetensors"]:
        kwargs["use_safetensors"] = True
    if cfg["device_map"]:
        kwargs["device_map"] = "cuda"

    import os as _os
    _os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    _os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

    pipe = DiffusionPipeline.from_pretrained(cfg["repo"], **kwargs)

    if not cfg["device_map"]:
        pipe = pipe.to("cuda")

    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    pipe.vae.to(torch.float16)
    torch.cuda.empty_cache()

    _pipe_cache["key"]  = model_key
    _pipe_cache["pipe"] = pipe
    return pipe



def _ollama_running_models() -> list[str]:
    """Return the names of models currently loaded in Ollama (via /api/ps)."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/ps", timeout=5)
        if r.ok:
            return [m.get("name", "") for m in r.json().get("models", []) if m.get("name")]
    except Exception:
        pass
    return []


def _unload_model(name: str, timeout: int = 120) -> bool:
    """Send keep_alive=0 to Ollama for *name* and wait up to *timeout* seconds.

    Returns True if the model is gone from /api/ps afterwards, False otherwise.
    Uses /api/chat (more reliable across Ollama versions) with keep_alive=0.
    """
    try:
        requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": name, "messages": [], "keep_alive": 0},
            timeout=timeout,
        )
    except Exception:
        pass

    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/ps", timeout=5)
        if r.ok:
            still_loaded = [m.get("name") for m in r.json().get("models", [])]
            return name not in still_loaded
    except Exception:
        pass
    return True


def _unload_ollama_models(except_model: str = "", timeout: int = 120):
    """Unload all running Ollama models from VRAM (optionally skip *except_model*).

    Uses a generous timeout so large models have time to fully unload before
    the next model is loaded — the original 10-second timeout was the root
    cause of Ollama freezing when switching between heavy models.
    """
    for name in _ollama_running_models():
        if name and name != except_model:
            _unload_model(name, timeout=timeout)


def _run_generation(job_id: str, params: dict):
    """Background worker — runs in a thread, updates _img_jobs."""
    import time as _time, io, warnings
    warnings.filterwarnings("ignore")

    job = _img_jobs[job_id]
    job["status"]  = "running"
    job["message"] = "Acquiring GPU lock…"

    with _img_lock:
        try:
            job["message"] = f"Loading model ({params['model']})…"
            pipe = _load_pipeline(params["model"])

            seed = params.get("seed", 42)
            if seed < 0:
                import random
                seed = random.randint(0, 2**31 - 1)

            total_steps = params.get("steps", 30)
            job.update(total_steps=total_steps, step=0,
                       message=f"Generating… step 0/{total_steps}")
            t0 = _time.time()

            generator = torch.Generator(device="cuda").manual_seed(seed)

            def _step_cb_new(pipe_obj, i, t, kwargs):
                job["step"] = i + 1
                job["message"] = f"Generating… step {i+1}/{total_steps}"
                return kwargs

            def _step_cb_old(i, t, latents):
                job["step"] = i + 1
                job["message"] = f"Generating… step {i+1}/{total_steps}"

            gen_kw = dict(
                prompt=params["prompt"],
                negative_prompt=params.get("negative_prompt", ""),
                width=params["width"], height=params["height"],
                num_inference_steps=total_steps,
                guidance_scale=params.get("guidance_scale", 7.0),
                generator=generator,
            )
            try:
                image = pipe(**gen_kw, callback_on_step_end=_step_cb_new).images[0]
            except TypeError:
                image = pipe(**gen_kw, callback=_step_cb_old, callback_steps=1).images[0]

            elapsed = round(_time.time() - t0, 1)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            IMAGEGEN_SAVE_DIR.mkdir(parents=True, exist_ok=True)
            fname = f"{int(_time.time())}_{seed}_{params['model']}.png"
            (IMAGEGEN_SAVE_DIR / fname).write_bytes(buf.getvalue())

            job["status"]    = "done"
            job["image_b64"] = b64
            job["elapsed"]   = elapsed
            job["seed"]      = seed
            job["filename"]  = fname
            job["message"]   = "Done"

        except Exception as exc:
            job["status"] = "error"
            job["error"]  = str(exc)


@app.route("/imagegen")
def imagegen_page():
    if not session.get("authenticated"):
        return jsonify({"error": "Unauthorized"}), 401
    return render_template("imagegen.html")


@app.route("/api/imagegen", methods=["POST"])
@login_required
def api_imagegen_start():
    if not _DIFFUSERS_OK:
        return jsonify({"error": "Image generation not available (diffusers not installed)."}), 503

    data = request.get_json(force=True) or {}

    model = data.get("model", "anime")
    if model not in IMAGEGEN_MODELS:
        return jsonify({"error": f"Unknown model '{model}'."}), 400

    w = int(data.get("width",  824))
    h = int(data.get("height", 824))
    w = min(max(8, (w // 8) * 8), 824)
    h = min(max(8, (h // 8) * 8), 824)
    if (w, h) not in SAFE_RESOLUTIONS:
        best = min(SAFE_RESOLUTIONS, key=lambda r: abs(r[0]-w) + abs(r[1]-h))
        w, h = best

    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    params = {
        "model":           model,
        "prompt":          prompt,
        "negative_prompt": data.get("negative_prompt", ""),
        "width":           w,
        "height":          h,
        "steps":           max(10, min(60, int(data.get("steps", 30)))),
        "guidance_scale":  float(data.get("guidance_scale", 7.0)),
        "seed":            int(data.get("seed", 42)),
    }

    threading.Thread(target=_unload_ollama_models, daemon=True).start()

    job_id = str(uuid.uuid4())
    _img_jobs[job_id] = {"status": "pending", "message": "Queued — freeing Ollama VRAM…"}

    t = threading.Thread(target=_run_generation, args=(job_id, params), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/api/imagegen/<job_id>")
@login_required
def api_imagegen_poll(job_id):
    job = _img_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job ID"}), 404
    resp = {
        "status":  job["status"],
        "message": job.get("message", ""),
    }
    if job["status"] == "done":
        resp["image_b64"] = job.get("image_b64", "")
        resp["elapsed"]   = job.get("elapsed", 0)
        resp["seed"]      = job.get("seed", 0)
        del _img_jobs[job_id]
    elif job["status"] == "error":
        resp["error"] = job.get("error", "Unknown error")
        del _img_jobs[job_id]
    return jsonify(resp)

@app.route("/api/imagegen/gallery")
@login_required
def api_imagegen_gallery_list():
    IMAGEGEN_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    imgs = []
    for f in sorted(IMAGEGEN_SAVE_DIR.glob("*.png"), reverse=True):
        s = f.stat()
        imgs.append({"filename": f.name,
                     "url": f"/api/imagegen/gallery/img/{f.name}",
                     "size": s.st_size, "created": s.st_mtime})
    return jsonify({"images": imgs})


@app.route("/api/imagegen/gallery/img/<filename>")
@login_required
def api_imagegen_gallery_image(filename):
    if not filename.endswith(".png") or "/" in filename or ".." in filename:
        return jsonify({"error": "Invalid"}), 400
    return send_from_directory(str(IMAGEGEN_SAVE_DIR.resolve()), filename)


@app.route("/api/imagegen/gallery/img/<filename>", methods=["DELETE"])
@login_required
def api_imagegen_gallery_delete(filename):
    if not filename.endswith(".png") or "/" in filename or ".." in filename:
        return jsonify({"error": "Invalid"}), 400
    fpath = IMAGEGEN_SAVE_DIR / filename
    if fpath.exists():
        fpath.unlink()
        return jsonify({"ok": True})
    return jsonify({"error": "Not found"}), 404


@app.route("/api/imagegen/stream/<job_id>")
@login_required
def api_imagegen_stream(job_id):
    """SSE stream — keeps tunnel alive during long generations, sends live step progress."""
    def _sse():
        last_step, last_status = -1, ""
        while True:
            job = _img_jobs.get(job_id)
            if not job:
                yield "data: " + __import__("json").dumps({"error": "Job not found"}) + "\n\n"
                return
            status = job["status"]
            step   = job.get("step", 0)
            total  = job.get("total_steps", 0)
            msg    = job.get("message", "")
            if step != last_step or status != last_status:
                last_step, last_status = step, status
                yield ("data: " + __import__("json").dumps(
                    {"status": status, "message": msg, "step": step, "total_steps": total}
                ) + "\n\n")
                if status == "done":
                    yield ("data: " + __import__("json").dumps({
                        "status":    "done",
                        "image_b64": job.get("image_b64", ""),
                        "elapsed":   job.get("elapsed", 0),
                        "seed":      job.get("seed", 0),
                        "filename":  job.get("filename", ""),
                    }) + "\n\n")
                    _img_jobs.pop(job_id, None)
                    return
                elif status == "error":
                    yield ("data: " + __import__("json").dumps(
                        {"status": "error", "error": job.get("error", "")}
                    ) + "\n\n")
                    _img_jobs.pop(job_id, None)
                    return
            else:
                yield ": keep-alive\n\n"
            time.sleep(0.4)
    return Response(stream_with_context(_sse()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

import subprocess, shutil, pty, fcntl, termios, select, struct

def _sandbox_path(rel: str) -> Path:
    """Resolve a relative path inside SANDBOX_DIR; raise ValueError if it escapes."""
    cleaned = rel.lstrip("/")
    p = (SANDBOX_DIR / cleaned).resolve()
    if not str(p).startswith(str(SANDBOX_DIR)):
        raise ValueError("Path escapes sandbox")
    return p

_SESSIONS: dict = {}
_SESSIONS_LOCK = threading.Lock()

_SAFE_ENV = {
    "PATH":    "/usr/local/bin:/usr/bin:/bin",
    "HOME":    str(SANDBOX_DIR),
    "TMPDIR":  str(SANDBOX_DIR / "tmp"),
    "SANDBOX": str(SANDBOX_DIR),
    "TERM":    "xterm-256color",
    "LANG":    "en_US.UTF-8",
    "PS1":     r"sandbox:\w \$ ",
}

class PtySession:
    """A bash process running inside SANDBOX_DIR via a PTY."""

    IDLE_TIMEOUT = 900

    def __init__(self, sid: str):
        self.sid        = sid
        self.buf        = bytearray()
        self.buf_lock   = threading.Lock()
        self.last_read  = time.time()
        self.alive      = True
        self.exit_code  = None

        (SANDBOX_DIR / "tmp").mkdir(exist_ok=True)

        self.master_fd, slave_fd = pty.openpty()
        self._set_winsize(self.master_fd, 24, 120)

        self.proc = subprocess.Popen(
            ["/bin/bash", "--norc", "--noprofile"],
            stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
            cwd=str(SANDBOX_DIR),
            env=_SAFE_ENV,
            close_fds=True,
            preexec_fn=os.setsid,
        )
        os.close(slave_fd)

        fl = fcntl.fcntl(self.master_fd, fcntl.F_GETFL)
        fcntl.fcntl(self.master_fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog.start()

    @staticmethod
    def _set_winsize(fd, rows, cols):
        try:
            s = struct.pack("HHHH", rows, cols, 0, 0)
            fcntl.ioctl(fd, termios.TIOCSWINSZ, s)
        except Exception:
            pass

    def _read_loop(self):
        while self.alive:
            try:
                r, _, _ = select.select([self.master_fd], [], [], 0.05)
                if r:
                    data = os.read(self.master_fd, 4096)
                    if not data:
                        break
                    with self.buf_lock:
                        self.buf.extend(data)
                        if len(self.buf) > 512 * 1024:
                            self.buf = self.buf[-256 * 1024:]
            except (OSError, ValueError):
                break
        self.alive = False
        if self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.exit_code = self.proc.wait()

    def _watchdog_loop(self):
        while self.alive:
            time.sleep(30)
            if time.time() - self.last_read > self.IDLE_TIMEOUT:
                self.kill()

    def read(self) -> bytes:
        self.last_read = time.time()
        with self.buf_lock:
            data = bytes(self.buf)
            self.buf.clear()
        return data

    def write(self, data: bytes):
        if not self.alive:
            return
        try:
            os.write(self.master_fd, data)
        except OSError:
            self.alive = False

    def resize(self, rows: int, cols: int):
        self._set_winsize(self.master_fd, rows, cols)

    def kill(self):
        self.alive = False
        try:
            os.close(self.master_fd)
        except Exception:
            pass
        if self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        with _SESSIONS_LOCK:
            _SESSIONS.pop(self.sid, None)

@app.route("/api/terminal/session/create", methods=["POST"])
@login_required
def api_term_create():
    user_id = session.get("user_id") or "anon"
    sid = f"sess_{user_id}"
    with _SESSIONS_LOCK:
        existing = _SESSIONS.get(sid)
        if existing and not existing.alive:
            existing.kill()
            existing = None
        if not existing:
            s = PtySession(sid)
            _SESSIONS[sid] = s
    return jsonify({"sid": sid, "ok": True})


@app.route("/api/terminal/session/write", methods=["POST"])
@login_required
def api_term_write():
    data = request.get_json(silent=True) or {}
    sid  = data.get("sid", "")
    text = data.get("text", "")
    with _SESSIONS_LOCK:
        s = _SESSIONS.get(sid)
    if not s or not s.alive:
        return jsonify({"error": "session not found"}), 404
    s.write(text.encode("utf-8", errors="replace"))
    return jsonify({"ok": True})


@app.route("/api/terminal/session/read")
@login_required
def api_term_read():
    sid     = request.args.get("sid", "")
    timeout = min(float(request.args.get("timeout", "25")), 30)
    with _SESSIONS_LOCK:
        s = _SESSIONS.get(sid)
    if not s:
        return jsonify({"data": "", "alive": False})

    deadline = time.time() + timeout
    while time.time() < deadline:
        raw = s.read()
        if raw:
            return jsonify({"data": raw.decode("utf-8", errors="replace"), "alive": s.alive})
        if not s.alive:
            break
        time.sleep(0.05)

    return jsonify({"data": "", "alive": s.alive})


@app.route("/api/terminal/session/resize", methods=["POST"])
@login_required
def api_term_resize():
    data = request.get_json(silent=True) or {}
    sid  = data.get("sid", "")
    rows = int(data.get("rows", 24))
    cols = int(data.get("cols", 120))
    with _SESSIONS_LOCK:
        s = _SESSIONS.get(sid)
    if s:
        s.resize(rows, cols)
    return jsonify({"ok": True})


@app.route("/api/terminal/session/kill", methods=["POST"])
@login_required
def api_term_kill():
    data = request.get_json(silent=True) or {}
    sid  = data.get("sid", "")
    with _SESSIONS_LOCK:
        s = _SESSIONS.pop(sid, None)
    if s:
        s.kill()
    return jsonify({"ok": True})

@app.route("/api/terminal/files")
@login_required
def api_terminal_files():
    rel = request.args.get("path", "").strip()
    try:
        target = _sandbox_path(rel) if rel else SANDBOX_DIR
        if not target.is_dir():
            target = SANDBOX_DIR
    except ValueError:
        target = SANDBOX_DIR

    entries = []
    try:
        for item in sorted(target.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
            if item.name.startswith("."):
                continue
            entries.append({
                "name": item.name,
                "type": "dir" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None,
            })
    except Exception:
        pass

    rel_display = str(target.relative_to(SANDBOX_DIR)) if target != SANDBOX_DIR else ""
    return jsonify({"path": rel_display, "entries": entries})


@app.route("/api/terminal/read")
@login_required
def api_terminal_read():
    rel = request.args.get("path", "").strip()
    try:
        target = _sandbox_path(rel)
    except ValueError:
        return jsonify({"error": "Path escapes sandbox"}), 400
    if not target.is_file():
        return jsonify({"error": "Not a file"}), 404
    if target.stat().st_size > 512 * 1024:
        return jsonify({"error": "File too large (>512 KB)"}), 413
    try:
        return jsonify({"content": target.read_text(errors="replace")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/terminal/write", methods=["POST"])
@login_required
def api_terminal_write():
    data    = request.get_json(silent=True) or {}
    rel     = (data.get("path") or "").strip()
    content = data.get("content", "")
    try:
        target = _sandbox_path(rel)
    except ValueError:
        return jsonify({"error": "Path escapes sandbox"}), 400
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return jsonify({"ok": True})

#  TTS  (espeak-ng — local, no internet needed)
_TTS_VOICES = [
    ("en",          "English — Default",        "neutral"),
    ("en+m1",       "Male 1 — Light",           "male"),
    ("en+m2",       "Male 2",                   "male"),
    ("en+m3",       "Male 3",                   "male"),
    ("en+m7",       "Male 7 — Deep",            "male"),
    ("en+f1",       "Female 1",                 "female"),
    ("en+f2",       "Female 2",                 "female"),
    ("en+f3",       "Female 3",                 "female"),
    ("en+f4",       "Female 4 — Soft",          "female"),
    ("en-us",       "American English",         "neutral"),
    ("en-us+m3",    "American Male",            "male"),
    ("en-us+f3",    "American Female",          "female"),
    ("en-gb",       "British English",          "neutral"),
    ("en-gb+f4",    "British Female",           "female"),
    ("en-gb+m5",    "British Male",             "male"),
    ("en-sc",       "Scottish English",         "neutral"),
    ("en-au",       "Australian English",       "neutral"),
]

def _espeak_available():
    return shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None

def _espeak_cmd():
    return "espeak-ng" if shutil.which("espeak-ng") else "espeak"


@app.route("/api/tts/voices")
@login_required
def api_tts_voices():
    if not _espeak_available():
        return jsonify({"error": "espeak-ng not installed. Run: sudo apt install espeak-ng"}), 503
    voices = [{"id": v[0], "name": v[1], "gender": v[2]} for v in _TTS_VOICES]
    return jsonify({"voices": voices})


@app.route("/api/tts", methods=["POST"])
@login_required
def api_tts():
    """Generate speech via espeak-ng and return WAV audio."""
    if not _espeak_available():
        return jsonify({"error": "espeak-ng not installed. Run: sudo apt install espeak-ng"}), 503

    data    = request.get_json(silent=True) or {}
    text    = (data.get("text") or "").strip()[:4000]
    voice   = data.get("voice", "en") or "en"
    rate    = max(0.5, min(2.0, float(data.get("rate", 1.0))))
    speed   = int(90 + (rate - 0.5) * (350 - 90) / 1.5)
    pitch_f = max(0.5, min(2.0, float(data.get("pitch", 1.0))))
    pitch   = int(20 + (pitch_f - 0.5) * (90 - 20) / 1.5)

    if not text:
        return jsonify({"error": "No text provided"}), 400

    import re as _re
    if not _re.match(r'^[a-zA-Z0-9\-+]+$', voice):
        voice = "en"

    try:
        cmd = [_espeak_cmd(), "-v", voice, "-s", str(speed), "-p", str(pitch),
               "--stdout", text]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            cmd[2] = "en"
            result = subprocess.run(cmd, capture_output=True, timeout=30)
        if not result.stdout:
            return jsonify({"error": "espeak produced no audio"}), 500
        return Response(result.stdout, mimetype="audio/wav",
                        headers={"Cache-Control": "no-cache"})
    except subprocess.TimeoutExpired:
        return jsonify({"error": "TTS timed out"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/proxy_image")
@login_required
def proxy_image():
    """Proxy an external image through the server to avoid CORS/mixed-content issues."""
    img_url = request.args.get("url", "").strip()
    if not img_url or not img_url.startswith(("http://", "https://")):
        return jsonify({"error": "Invalid URL"}), 400
    try:
        r = requests.get(img_url, timeout=10,
                         headers={"User-Agent": "Mozilla/5.0 (compatible; OllamaGate/1.0)"},
                         stream=True)
        r.raise_for_status()
        ct = r.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
        if not ct.startswith("image/"):
            return jsonify({"error": "Not an image"}), 400
        data = b""
        for chunk in r.iter_content(8192):
            data += chunk
            if len(data) > 4 * 1024 * 1024:
                break
        return Response(data, mimetype=ct,
                        headers={"Cache-Control": "public, max-age=3600"})
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/proxy_video")
@login_required
def proxy_video():
    """Proxy an external video through the server for in-chat playback."""
    vid_url = request.args.get("url", "").strip()
    if not vid_url or not vid_url.startswith(("http://", "https://")):
        return jsonify({"error": "Invalid URL"}), 400
    try:
        range_header = request.headers.get("Range", None)
        req_headers = {"User-Agent": "Mozilla/5.0 (compatible; OllamaGate/1.0)"}
        if range_header:
            req_headers["Range"] = range_header
        r = requests.get(vid_url, timeout=20,
                         headers=req_headers, stream=True)
        r.raise_for_status()
        ct = r.headers.get("Content-Type", "video/mp4").split(";")[0].strip()
        if not ct.startswith("video/") and not ct.startswith("application/"):
            return jsonify({"error": "Not a video"}), 400
        status = r.status_code
        resp_headers = {
            "Content-Type": ct,
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600",
        }
        for h in ("Content-Range", "Content-Length"):
            if h in r.headers:
                resp_headers[h] = r.headers[h]

        def stream():
            for chunk in r.iter_content(65536):
                yield chunk

        return Response(stream_with_context(stream()), status=status,
                        headers=resp_headers)
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/manifest.json")
def manifest():
    return send_from_directory("static","manifest.json")

@app.route("/sw.js")
def service_worker():
    return send_from_directory("static","sw.js",mimetype="application/javascript")

def _expiry_watcher():
    while True:
        time.sleep(10)
        if _state["expires_at"] and _now() >= _state["expires_at"]:
            with _state_lock:
                print("\n[OllamaGate] ⏰ Token expired. All sessions revoked.")
                _state.update({"token":None,"expires_at":None,"connected_ips":{}})

def _do_setup(secs: int):
    """Apply a validated duration, generate a token, and print startup info."""
    token      = generate_token()
    expires_at = _now() + timedelta(seconds=secs)
    with _state_lock:
        _state["token"]      = token
        _state["expires_at"] = expires_at
    h, r = divmod(secs, 3600); m = r // 60
    print("\n" + "─"*55)
    print(f"  ✅  Token valid for: {'{}h {}m'.format(h,m) if h else '{}m'.format(m)}")
    print(f"\n  🔑  TOKEN:\n\n      {token}")
    print(f"\n  ⏰  Expires: {expires_at.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  📊  Stats:  http://0.0.0.0:{os.getenv('PORT',8080)}/stats")
    print(f"  📂  Benchmarks: {Path(BENCHMARK_DIR).resolve()}")
    print(f"  🔧  Tools enabled: web_search, web_fetch, calculator, datetime, save_memory")
    print(f"  📦  Sandbox:    {SANDBOX_DIR}")
    print("─"*55+"\n")


def interactive_setup():
    print("\n" + "═"*55)
    print("  🔐  OllamaGate — Enhanced Ollama Interface")
    print("═"*55)

    env_dur = os.getenv("DURATION", "").strip()
    if env_dur:
        try:
            secs = parse_duration(env_dur)
            print(f"\n  [auto] Using DURATION={env_dur} from environment.")
            _do_setup(secs)
            return
        except ValueError as e:
            print(f"  ⚠  DURATION env var invalid ({e}). Falling back to prompt.")

    while True:
        try:
            raw = input("\nHow long should the site be active? (e.g. '1 h' or '30 m'): ").strip()
        except (EOFError, OSError):
            print("\n  ⚠  No DURATION env var set and stdin is unavailable.")
            print("  ℹ  Defaulting to 8 hours. Set DURATION=Xh to customise.")
            raw = "8h"
        try:
            secs = parse_duration(raw)
            break
        except ValueError as e:
            print(f"  ⚠  {e}")

    _do_setup(secs)

Path(BENCHMARK_DIR).mkdir(parents=True, exist_ok=True)
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
(SANDBOX_DIR / "tmp").mkdir(exist_ok=True)
init_db()
interactive_setup()
threading.Thread(target=_expiry_watcher, daemon=True).start()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    print(f"[OllamaGate] Running → http://0.0.0.0:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)