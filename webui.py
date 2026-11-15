"""Helpers that embed media directly in the page (no reliance on Streamlit's media server)."""
import base64

MAX_EMBED_VIDEO_BYTES = 25_000_000


def png_html(png: bytes, alt: str = "") -> str:
    b64 = base64.b64encode(png).decode()
    return f'<img alt="{alt}" style="width:100%;height:auto" src="data:image/png;base64,{b64}">'


def video_html(mp4: bytes) -> str:
    b64 = base64.b64encode(mp4).decode()
    return (f'<video controls playsinline style="width:100%;max-height:70vh" '
            f'src="data:video/mp4;base64,{b64}"></video>')
