"""
Corporate Intelligence Engine - Streamlit Frontend

An interactive web UI for the corporate-intelligence-engine that connects to the
FastAPI backend and provides real-time visualization of agent reasoning steps.

Features:
  - Chat interface for user queries
  - Real-time agent activity visualization
  - Structured report display
  - Execution time tracking
  - Error handling and recovery

Run with:
  streamlit run frontend.py
"""

import streamlit as st
import re
import requests
import json
import os
from datetime import datetime, timedelta
from typing import Optional
import time

def _format_bullet_points(text: str) -> str:
    """Ensure bullet points (•) are on separate lines.
    
    Split on bullets and rebuild with double newlines (markdown paragraph breaks).
    """
    # Split the text on bullet characters
    parts = text.split('•')
    
    # First part (before any bullet) doesn't need a bullet prefix
    result = [parts[0].strip()]
    
    # Each subsequent part gets a bullet prefix on its own line
    for part in parts[1:]:
        stripped = part.strip()
        if stripped:
            result.append('• ' + stripped)
    
    # Join with double newlines (paragraph breaks in markdown)
    return '\n\n'.join(result)


def _fix_concatenated_text(text: str) -> str:
    """Fix common word concatenation issues from LLM output.
    
    Detects patterns where words are joined without spaces or have excessive spaces.
    """
    
    # First pass: Fix over-spaced capital letters (ticker symbols like "G O O G L")
    text = re.sub(r'\b([A-Z])\s+([A-Z])\s+([A-Z])\s+([A-Z])\s+([A-Z])\b', r'\1\2\3\4\5', text)
    text = re.sub(r'\b([A-Z])\s+([A-Z])\s+([A-Z])\s+([A-Z])\b', r'\1\2\3\4', text)
    text = re.sub(r'\b([A-Z])\s+([A-Z])\s+([A-Z])\b', r'\1\2\3', text)
    text = re.sub(r'\b([A-Z])\s+([A-Z])\b', r'\1\2', text)
    
    # Second pass: Fix common multi-word financial concatenations
    replacements = {
        r'(?<!\w)analysttarget(?!\w)': 'analyst target',
        r'(?<!\w)analystpoint(?!\w)': 'analyst point',
        r'(?<!\w)targetprice(?!\w)': 'target price',
        r'(?<!\w)targetpoint(?!\w)': 'target point',
        r'(?<!\w)thestocksis(?!\w)': 'the stock is',
        r'(?<!\w)thestocksits(?!\w)': 'the stock sits',
        r'(?<!\w)thestockcontinues(?!\w)': 'the stock continues',
        r'(?<!\w)stockcontinues(?!\w)': 'stock continues',
        r'(?<!\w)stocktrades(?!\w)': 'stock trades',
        r'(?<!\w)marginlift(?!\w)': 'margin lift',
        r'(?<!\w)margincompression(?!\w)': 'margin compression',
        r'(?<!\w)marginexpansion(?!\w)': 'margin expansion',
        r'(?<!\w)earningsacceleration(?!\w)': 'earnings acceleration',
        r'(?<!\w)earningsgrowth(?!\w)': 'earnings growth',
        r'(?<!\w)earningsdeceleration(?!\w)': 'earnings deceleration',
        r'(?<!\w)operatingleverage(?!\w)': 'operating leverage',
        r'(?<!\w)cashflow(?!\w)': 'cash flow',
        r'(?<!\w)freecash(?!\w)': 'free cash',
        r'(?<!\w)incrementalmargin(?!\w)': 'incremental margin',
        r'(?<!\w)incrementalrevenue(?!\w)': 'incremental revenue',
        r'(?<!\w)incrementalprofit(?!\w)': 'incremental profit',
        r'(?<!\w)valuationdisconnect(?!\w)': 'valuation disconnect',
        r'(?<!\w)valuationmultiple(?!\w)': 'valuation multiple',
        r'(?<!\w)valuationstretch(?!\w)': 'valuation stretch',
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Third pass: Fix currency immediately followed by text
    text = re.sub(r'(\$[\d,\.]+)([a-z]+)', r'\1 \2', text, flags=re.IGNORECASE)
    
    # Fourth pass: Break up obvious concatenated lowercase words
    # Insert spaces before common word prefixes within long lowercase runs
    common_starts = [
        'able', 'and', 'as', 'at', 'be', 'by', 'can', 'co', 'de', 'do', 'en', 'ex',
        'for', 'from', 'have', 'he', 'her', 'his', 'how', 'if', 'in', 'into', 'is',
        'it', 'just', 'like', 'may', 'me', 'more', 'most', 'my', 'no', 'not', 'of',
        'on', 'or', 'other', 'our', 'out', 'over', 'per', 'pre', 're', 'share',
        'so', 'the', 'their', 'then', 'there', 'these', 'they', 'this', 'those',
        'to', 'too', 'under', 'up', 'use', 'was', 'we', 'what', 'when', 'where',
        'which', 'who', 'why', 'with', 'would', 'yet', 'you', 'your'
    ]
    
    for word in common_starts:
        # Only in obviously concatenated contexts (long runs of lowercase)
        pattern = rf'([a-z]{4,})({word})(?=[a-z])'
        text = re.sub(pattern, rf'\1 \2', text, flags=re.IGNORECASE)
    
    # Fourth-B: Fix special case: )and followed by capital letter like ")andTSLA"
    text = re.sub(r'\)and([A-Z])', r') and \1', text)
    # Fix: number)and like "(10)and"
    text = re.sub(r'(\d)\)and([A-Z])', r'\1) and \2', text)
    
    # Fifth pass: Insert space between runs of lowercase and numbers
    # This helps with patterns like "thestocksits21" → "thestocksits 21"
    text = re.sub(r'([a-z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-z])', r'\1 \2', text)
    
    # Sixth pass: Ensure proper line breaks around analysis section headers
    # Pattern: bold emoji text followed by bullet should have line break
    text = re.sub(r'(\*\*[🔴🟢].*?— Round \d+\*\*)•', r'\1\n\n•', text)
    
    # Pattern: Synthesis section followed by more text should have line break
    text = re.sub(r'(\*\*Synthesis:\*\*[^•]+)(•|---|\*\*)', r'\1\n\n\2', text)
    
    return text


def _escape_currency(text: str) -> str:
    """Replace currency signs with \\$ to prevent LaTeX math mode interpretation.
    First unescapes HTML entities (&#36; → $), then escapes for LaTeX safety.
    Skips markdown table rows (lines starting with |).
    Also reformats raw market-cap integers like $1477689606144 → $1.48T."""
    import html


def _escape_currency(text: str) -> str:
    """Replace currency signs with \\$ to prevent LaTeX math mode interpretation.
    First unescapes HTML entities (&#36; → $), then escapes for LaTeX safety.
    Skips markdown table rows (lines starting with |).
    Also reformats raw market-cap integers like $1477689606144 → $1.48T."""
    import html

    def _fmt_large(m):
        n = int(m.group(1))
        if n >= 1_000_000_000_000:
            return f"${n/1_000_000_000_000:.2f}T"
        if n >= 1_000_000_000:
            return f"${n/1_000_000_000:.2f}B"
        if n >= 1_000_000:
            return f"${n/1_000_000:.2f}M"
        return f"${n:,}"

    # First: unescape any HTML entities (e.g., &#36; → $)
    text = html.unescape(text)

    lines = text.split('\n')
    result = []
    for line in lines:
        if line.strip().startswith('|'):
            result.append(line)  # table row — leave as-is
        else:
            # Fix malformed currency: backslash without dollar (e.g., \432.29 → $432.29)
            line = re.sub(r'\\(\d)', r'$\1', line)
            # Large raw integers first (market cap style: $1477689606144)
            line = re.sub(r'\$(\d{7,})', _fmt_large, line)
            # First: unescape any remaining HTML entities (e.g., &#36; → $)
            line = html.unescape(line)
            result.append(line)
    return '\n'.join(result)


def _stream_report_chunks(text: str):
    """Generator that yields short segments of text for streaming display.
    
    Chunks by sentences and short segments for fast, subtle streaming effect.
    """
    # Split on sentence boundaries (periods followed by space) but preserve formatting
    segments = []
    current_segment = ""
    
    for line in text.split('\n'):
        if not line.strip():
            # Empty line - yield what we have and empty line
            if current_segment:
                segments.append(current_segment + '\n\n')
                current_segment = ""
            else:
                segments.append('\n')
        else:
            # For lines with bullets or headers, keep them together
            if line.strip().startswith(('•', '-', '**', '##', '###')):
                if current_segment:
                    segments.append(current_segment)
                    current_segment = ""
                segments.append(line + '\n')
            else:
                # Split long lines into sentences
                sentences = re.split(r'(\. )', line)
                for i, part in enumerate(sentences):
                    current_segment += part
                    # Yield after complete sentence
                    if part == '. ' and i < len(sentences) - 1:
                        segments.append(current_segment)
                        current_segment = ""
                
                # Add newline after line if not empty
                if current_segment:
                    current_segment += '\n'
                elif line.strip():
                    segments.append(line + '\n')
    
    # Yield any remaining segment
    if current_segment:
        segments.append(current_segment)
    
    # Yield segments with small delay for subtle streaming effect
    for segment in segments:
        if segment.strip():
            yield segment
            time.sleep(0.01)  # 10ms delay between segments


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Corporate Intelligence Engine",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern CSS for enhanced UI/UX - Alibaba Cloud Console Aesthetic
st.markdown("""
<style>
    /* Color Palette */
    :root {
        --primary-purple: #667eea;
        --primary-dark-purple: #764ba2;
        --alibaba-orange: #ff6a00;
        --alibaba-light-orange: #FF8C42;
        --success: #10b981;
        --error: #ef4444;
        --clean-white: #ffffff;
        --light-gray: #f8fafc;
        --border-gray: #e2e8f0;
        --text-dark: #0f172a;
        --text-muted: #64748b;
    }
    
    /* Typography & General */
    * {
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }
    
    h1, h2, h3 {
        font-weight: 600;
        letter-spacing: -0.5px;
        color: var(--text-dark);
    }
    
    h1 { font-size: 1.5em; }
    h2 { font-size: 1.35em; }
    h3 { font-size: 1.05em; }
    
    /* Main Container - Clean light gray background */
    .main {
        background: var(--light-gray);
    }
    
    /* Header & Title - Minimalist card with top accent */
    /* Header Container - Compact and tight */
    .header-container {
        padding: 12px 24px;
        margin-bottom: 14px;
        background: var(--clean-white);
        border: 1px solid var(--border-gray);
        border-top: 4px solid var(--alibaba-orange);
        border-radius: 0 0 8px 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .main-title {
        text-align: center;
        color: var(--text-dark);
        margin: 0;
        font-size: 1.9em;
        font-weight: 700;
        line-height: 1.2;
    }
    
    /* Cards & Boxes */
    .card {
        background: var(--clean-white);
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
        border: 1px solid var(--border-gray);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 15px;
    }
    
    .card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transform: translateY(-1px);
    }
    
    /* Status Box - Clean white card with left accent border */
    .status-box {
        background: var(--clean-white);
        padding: 16px 20px;
        border-radius: 8px;
        border: 1px solid var(--border-gray);
        border-left: 4px solid var(--alibaba-orange);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        color: var(--text-dark);
    }
    
    .status-box * {
        color: var(--text-dark);
    }
    
    .status-box.success {
        border-left-color: var(--success);
    }
    
    .status-box.error {
        border-left-color: var(--error);
    }
    
    /* Orange badge/box text styling */
    .routing-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        margin: 5px 0;
        background: var(--alibaba-orange);
        color: white !important;
    }
    
    .research-badge {
        background: var(--alibaba-orange);
        color: white !important;
    }
    
    .general-badge {
        background: var(--alibaba-orange);
        color: white !important;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        margin: 5px 5px 5px 0;
        transition: all 0.3s ease;
    }
    
    .badge-success {
        background: var(--success);
        color: white;
    }
    
    .badge-error {
        background: var(--error);
        color: white;
    }
    
    .badge-research {
        background: #3b82f6;
        color: white;
    }
    
    .badge-general {
        background: #a855f7;
        color: white;
    }
    
    /* Metric Box - Clean white card with faint border */
    .metric-box {
        background: var(--clean-white);
        padding: 16px;
        border-radius: 8px;
        border: 1px solid var(--border-gray);
        text-align: center;
        margin: 8px 0;
        transition: all 0.3s ease;
    }
    
    .metric-box:hover {
        border-color: var(--alibaba-orange);
        box-shadow: 0 2px 8px rgba(255, 106, 0, 0.1);
    }
    
    .metric-value {
        font-size: 1.4em;
        font-weight: 700;
        color: var(--text-dark);
        margin: 8px 0 3px 0;
    }
    
    .metric-label {
        font-size: 0.9em;
        color: var(--text-muted);
        font-weight: 500;
    }
    
    /* Logs & Code */
    .log-entry {
        font-family: 'Fira Code', 'Monaco', monospace;
        font-size: 0.95em;
        line-height: 1.6;
        color: var(--text-dark);
        background: var(--light-gray);
        padding: 2px 4px;
        border-radius: 4px;
    }
    
    .log-container {
        background: var(--light-gray);
        border: 1px solid var(--border-gray);
        border-radius: 8px;
        padding: 12px;
        max-height: 400px;
        overflow-y: auto;
    }
    
    /* Success & Error Badges */
    .success-badge {
        color: var(--success);
        font-weight: 600;
    }
    
    .error-badge {
        color: var(--error);
        font-weight: 600;
    }
    
    /* Tables */
    table {
        border-collapse: collapse;
        width: 100%;
        font-size: 1.0em;
    }
    
    th {
        background-color: #f1f5f9;
        color: #475569;
        padding: 8px;
        text-align: left;
        font-weight: 600;
        border-bottom: 2px solid var(--border-gray);
        text-transform: uppercase;
        font-size: 0.72rem;
        letter-spacing: 0.5px;
    }
    
    td {
        padding: 12px;
        border-bottom: 1px solid var(--border-gray);
        color: var(--text-dark);
    }
    
    tr:hover {
        background-color: var(--light-gray);
    }
    
    /* Buttons - Alibaba Orange */
    button {
        background: var(--alibaba-orange);
        color: white !important;
        border: none;
        padding: 10px 20px;
        border-radius: 6px;
        cursor: pointer;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    button:hover {
        box-shadow: 0 4px 12px rgba(255, 106, 0, 0.25);
        transform: translateY(-1px);
    }
    
    button * {
        color: white !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--clean-white);
        border-right: 1px solid var(--border-gray);
    }
    
    /* Input Fields */
    input, textarea, select {
        border: 1px solid var(--border-gray) !important;
        border-radius: 6px !important;
        padding: 10px !important;
        font-size: 0.95em !important;
        background: var(--clean-white) !important;
        color: var(--text-dark) !important;
        transition: all 0.3s ease !important;
    }
    
    input:focus, textarea:focus, select:focus {
        border-color: var(--alibaba-orange) !important;
        box-shadow: 0 0 0 3px rgba(255, 106, 0, 0.1) !important;
    }
    
    /* Clean White Card for the Interface Prompt */
    .interface-box {
        background-color: #ffffff !important;
        padding: 16px 20px !important;
        border-radius: 8px !important;
        border: 1px solid var(--border-gray) !important;
        border-left: 4px solid var(--alibaba-orange) !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        margin-bottom: 20px !important;
    }
    
    .interface-title {
        color: var(--text-dark) !important;
        font-weight: 600 !important;
        font-size: 1.15rem !important;
        margin: 0 !important;
    }
    
    .interface-subtitle {
        color: var(--text-muted) !important;
        font-size: 0.9rem !important;
        margin: 4px 0 0 0 !important;
    }
    
    /* ─────────────────────────────────────────────────────────────────
       💥 ULTIMATE FIX: FORCE ALL INNER BUTTON LABELS TO DARK SLATE
       ───────────────────────────────────────────────────────────────── */
    
    /* Target the buttons AND every single element nested inside them */
    div[data-testid="stButton"] button,
    div[data-testid="stButton"] button *,
    button[data-testid="stBaseButton-secondary"],
    button[data-testid="stBaseButton-secondary"] * {
        color: #334155 !important;
        -webkit-text-fill-color: #334155 !important; /* Forces color past browser theme engine overrides */
        background-color: #ffffff !important;
    }

    /* Target the hover state for the buttons AND every nested element inside them */
    div[data-testid="stButton"] button:hover,
    div[data-testid="stButton"] button:hover *,
    button[data-testid="stBaseButton-secondary"]:hover,
    button[data-testid="stBaseButton-secondary"]:hover * {
        color: #ff6a00 !important; /* Turn text Alibaba Orange on hover */
        -webkit-text-fill-color: #ff6a00 !important;
        border-color: #ff6a00 !important;
    }

    /* Clean border wrapper around the button container */
    div[data-testid="stButton"] button, 
    button[data-testid="stBaseButton-secondary"] {
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }

    /* ─────────────────────────────────────────────────────────────────
       💥 ULTIMATE FIX: FORCE TOP DEPLOY TOOLBAR ELEMENTS TO DARK SLATE
       ───────────────────────────────────────────────────────────────── */
    
    /* Target the top layout header and every single child element nested within it */
    header[data-testid="stHeader"],
    header[data-testid="stHeader"] *,
    header[data-testid="stHeader"] div,
    header[data-testid="stHeader"] span,
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] a {
        color: #475569 !important;
        -webkit-text-fill-color: #475569 !important; /* Force color past browser/theme overrides */
    }

    /* Give the top header buttons a clean layout border profile if they render with backgrounds */
    header[data-testid="stHeader"] button {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 4px !important;
    }
    
    header[data-testid="stHeader"] button:hover {
        border-color: #ff6a00 !important;
        color: #ff6a00 !important;
        -webkit-text-fill-color: #ff6a00 !important;
    }
    
    /* Prevent stale element fading during long computations */
    [data-stale="true"] {
        opacity: 1 !important;
        filter: none !important;
        backdrop-filter: none !important;
    }
    
    [data-testid="stChatMessage"][data-stale="true"] {
        opacity: 1 !important;
        filter: none !important;
    }
    
    [data-stale="true"] [data-testid="stChatMessage"] {
        opacity: 1 !important;
        filter: none !important;
    }
    
    /* Chat message formatting */
    [data-testid="stChatMessage"] p {
        line-height: 1.6;
        margin: 8px 0;
        color: var(--text-dark);
    }
    
    [data-testid="stChatMessage"] ul {
        margin: 10px 0;
        padding-left: 24px;
    }
    
    [data-testid="stChatMessage"] li {
        margin: 6px 0;
        line-height: 1.6;
    }
    
    [data-testid="stChatMessage"] strong {
        display: inline;
        color: var(--text-dark);
    }
    
    [data-testid="stChatMessage"] {
        word-wrap: break-word;
        overflow-wrap: break-word;
        color: var(--text-dark);
    }
    
    [data-testid="stChatMessage"] a {
        color: var(--alibaba-orange);
        text-decoration: none;
    }
    
    [data-testid="stChatMessage"] a:hover {
        text-decoration: underline;
    }
    
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_response" not in st.session_state:
    st.session_state.last_response = None

if "api_available" not in st.session_state:
    st.session_state.api_available = None

if "approval_submitted" not in st.session_state:
    st.session_state.approval_submitted = False

if "approval_timeout" not in st.session_state:
    st.session_state.approval_timeout = None

if "pending_approval_request" not in st.session_state:
    st.session_state.pending_approval_request = None

if "trade_completed" not in st.session_state:
    st.session_state.trade_completed = False

if "last_completed_trade" not in st.session_state:
    st.session_state.last_completed_trade = None


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Backend routing: Try Streamlit secrets (Cloud), fall back to .env (Alibaba/Local)
try:
    API_BASE_URL = st.secrets.get("BACKEND_API_URL", "http://localhost:8002")
except (FileNotFoundError, AttributeError):
    # Suppress warning and use environment variables instead
    API_BASE_URL = os.getenv("BACKEND_API_URL", "http://localhost:8002")

# Clean up any accidental trailing slashes
API_BASE_URL = API_BASE_URL.rstrip("/")

API_HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
API_ANALYZE_ENDPOINT = f"{API_BASE_URL}/api/analyze"
API_ANALYZE_STREAM_ENDPOINT = f"{API_BASE_URL}/api/analyze/stream"
API_PORTFOLIO_ENDPOINT = f"{API_BASE_URL}/api/portfolio"
API_STATUS_ENDPOINT = f"{API_BASE_URL}/api/requests"
REQUEST_TIMEOUT = 600  # seconds (10 minutes max - increased for OAuth/MCP calls)
INACTIVITY_TIMEOUT = 180  # seconds (3 minutes of no log updates = stuck)
POLL_INTERVAL = 5  # seconds (poll every 5 seconds for new logs)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_api_health() -> bool:
    """
    Check if the FastAPI backend is running and healthy.
    Uses POST /invoke (verified working endpoint on Alibaba Cloud).
    
    Returns:
        bool: True if API is available, False otherwise
    """
    try:
        # Use POST to /invoke - the verified working endpoint on Alibaba Cloud
        # GET requests to root may trigger file download restrictions on some cloud providers
        # Timeout: 30s to allow cold starts on serverless (8-12s to wake up container)
        response = requests.post(f"{API_BASE_URL}/invoke", timeout=30)
        
        # Check for successful response with expected status
        if response.status_code == 200:
            data = response.json()
            return data.get("status") == "success"
    except Exception as e:
        return False
    return False


_LOG_SEPARATORS = frozenset({"=", "-", " ", "\n"})


def _extract_log_message(raw_line: str) -> tuple:
    """
    Parse a loguru line and return (level, message).
    Returns ("", "") for separator lines / empty messages.

    Input: "2026-06-30 23:06:21 | INFO     | module:func - message text"
    """
    level = "INFO"
    if "| WARNING " in raw_line or "| WARN " in raw_line:
        level = "WARNING"
    elif "| ERROR " in raw_line:
        level = "ERROR"

    msg = raw_line.split(" - ", 1)[1].strip() if " - " in raw_line else raw_line.strip()

    if not msg or set(msg) <= _LOG_SEPARATORS:
        return "", ""
    return level, msg


def _format_agent_step(level: str, msg: str) -> str:
    """
    Map a backend log message to a clean, icon-prefixed display line.
    Returns "" to skip the line entirely.
    """
    m = msg.strip()

    # ── Hard skips (internal plumbing / separators) ──────────────────────────
    _skip = (
        "BUILDING STATE GRAPH", "GRAPH COMPILATION",
        "Building state graph", "Executing orchestrator graph",
        "[ENTERING ", "[CONDITIONAL ROUTING]", "[CONDITIONAL EDGE]",
        "NODE COMPLETE", "[REPORTING NODE", "Report generation complete",
        "Using pre-generated report", "Streaming analysis started",
        "[EXTERNAL TOOL SUCCESS]", "Retrieved REAL data:",
        "[EXTERNAL TOOL] Fetching REAL",
        "Qwen Response:", "Qwen response received",
        "[COMMITTEE] Convening", "[COMMITTEE] Deliberation complete",
        "APPROVAL EXECUTION NODE", "NO EXECUTION",
        "preserving existing report",
        "## 🏛️ Investment Committee Verdict",  # Skip old verdict markdown
        "🟡 Final Verdict", "🟢 Final Verdict", "🔴 Final Verdict",  # Skip verdict lines
    )
    if any(s in m for s in _skip):
        return ""

    # ── Graph wiring details → collapse to one summary line ──────────────────
    # Individual ✓ messages from build_graph (nodes, edges, entry point) are
    # implementation noise. Suppress them all; "Graph compiled" emits the summary.
    if m.startswith("✓ ") or m.startswith("✓"):
        return ""   # suppressed — summary fires on "Graph compiled" below
    if "Graph compiled" in m:
        return "🔧 **Analysis pipeline ready**"

    # ── Errors / Warnings ────────────────────────────────────────────────────
    if level == "ERROR":
        return f"🔴 **Error:** {m}"
    if level == "WARNING" and "[CHECKPOINT]" not in m and "[SKIPPING]" not in m:
        return f"⚠️ {m}"

    # ── User query ───────────────────────────────────────────────────────────
    if m.startswith("User Input:"):
        return f"💬 **Query:** {m.split(':', 1)[1].strip()}"

    # ── Triage / routing ─────────────────────────────────────────────────────
    if "Invoking Qwen LLM for structured routing" in m:
        return "🧭 Determining intent and routing..."
    if "Extracted Ticker:" in m:
        return f"🎯 **Ticker identified:** {m.split(':', 1)[1].strip()}"
    if "Routing Decision:" in m:
        decision = m.split(":", 1)[1].strip()
        label = {"research": "Investment Research", "general_q": "General Q&A",
                 "trading": "Direct Trade", "portfolio": "Portfolio Analysis"}.get(decision.lower(), decision.capitalize())
        return f"🔀 **Routing →** {label}"
    if "Confidence:" in m:
        return f"📊 {m}"
    if "Reasoning:" in m:
        text = m.split(":", 1)[1].strip()
        return f"💭 _{text}_"

    # ── External data ─────────────────────────────────────────────────────────
    if "[EXTERNAL TOOL] Calling" in m or "Fetching" in m:
        return "📡 **Fetching live market data + fundamentals** (Yahoo Finance MCP + SEC EDGAR)..."
    if "[EXTERNAL TOOL RESULT]" in m:
        source = m.split("]", 1)[1].strip()
        return f"✅ **Data source:** {source}"
    if "[SEC] Loaded" in m:
        return f"📑 {m.split('] ', 1)[1].strip()}"
    if "[SEC] EDGAR fetch failed" in m or "[SEC] No 10-Q" in m:
        return f"⚠️ SEC filings unavailable — using market data only"
    if m.strip().startswith("Price:"):
        return f"&nbsp;&nbsp;&nbsp;💵 {m.strip()}"
    if m.strip().startswith("Change:"):
        return f"&nbsp;&nbsp;&nbsp;📈 {m.strip()}"
    if m.strip().startswith("Volume:"):
        return f"&nbsp;&nbsp;&nbsp;📊 {m.strip()}"
    if m.strip().startswith("P/E Ratio:"):
        return f"&nbsp;&nbsp;&nbsp;📐 {m.strip()}"
    if m.strip().startswith("EPS:"):
        return f"&nbsp;&nbsp;&nbsp;💰 {m.strip()}"
    if m.strip().startswith("Analyst Target:"):
        return f"&nbsp;&nbsp;&nbsp;🎯 {m.strip()}"
    if m.strip().startswith("Sector:"):
        return f"&nbsp;&nbsp;&nbsp;🏭 {m.strip()}"

    # ── LLM calls ─────────────────────────────────────────────────────────────
    if "Calling Qwen" in m and "structured output" in m:
        return "🤖 **Qwen LLM** thinking..."
    if "Invoking Qwen" in m:
        return "🤖 **Qwen LLM** processing..."

    # ── Investment Committee ──────────────────────────────────────────────────
    if "Convening Tri-Agent Investment Committee" in m:
        return "🏛️ **Convening Investment Committee** — Bull vs Bear vs Director"
    if "[COMMITTEE] Debate rounds:" in m:
        rounds = m.split(":", 1)[1].strip()
        return f"&nbsp;&nbsp;&nbsp;🔄 Running **{rounds}** debate rounds"
    if "[COMMITTEE] Bull Analyst delivered" in m:
        r = m.rsplit("round", 1)[-1].strip().split()[0]
        return f"&nbsp;&nbsp;&nbsp;🟢 **Bull Analyst** — Round {r} argument delivered"
    if "[COMMITTEE] Bear Auditor delivered" in m:
        r = m.rsplit("round", 1)[-1].strip().split()[0]
        return f"&nbsp;&nbsp;&nbsp;🔴 **Bear Auditor** — Round {r} argument delivered"
    if "[COMMITTEE] Director verdict:" in m:
        v = m.split(":", 1)[1].strip()
        return f"&nbsp;&nbsp;&nbsp;⚖️ **Director verdict:** {v}"
    if "Committee reached verdict:" in m:
        v = m.split(":", 1)[1].strip()
        return f"🏆 **Committee verdict:** {v}"
    if "[NO CHECKPOINT]" in m:
        return f"✅ {m.replace('[NO CHECKPOINT] ', '')}"
    
    # ── Skills ────────────────────────────────────────────────────────────────
    if "[SKILLS] Invoking Portfolio Analyzer" in m:
        return "🎯 **Portfolio Analyzer** — evaluating diversification & allocation"
    if "[SKILLS] Portfolio analysis:" in m:
        score = m.split("score", 1)[-1].strip()
        return f"&nbsp;&nbsp;&nbsp;📊 Portfolio: {score}"
    if "[SKILLS] Invoking Risk Manager" in m:
        return "⚠️ **Risk Manager** — assessing volatility & correlation"
    if "[SKILLS] Risk assessment:" in m:
        details = m.split(":", 1)[-1].strip()
        return f"&nbsp;&nbsp;&nbsp;📈 Risk: {details}"
    if "[SKILLS] Checking stop-loss/take-profit" in m:
        return "🔔 **Stop-Loss/Take-Profit Triggers** — monitoring thresholds"
    if "[SKILLS]" in m and "active trigger" in m:
        triggers = m.split("(")[1].split()[0] if "(" in m else "0"
        return f"&nbsp;&nbsp;&nbsp;🚨 {triggers} active trigger(s) detected"
    if "[SKILLS]" in m and ("Summary:" in m or "No active" in m):
        summary = m.split("Summary:", 1)[-1].strip() if "Summary:" in m else "Portfolio looks good"
        return f"&nbsp;&nbsp;&nbsp;💡 {summary}"

    # ── Human-in-loop checkpoint ──────────────────────────────────────────────
    if "[CHECKPOINT]" in m or "[SKIPPING]" in m:
        text = m.replace("[CHECKPOINT] ", "").replace("[SKIPPING] ", "")
        return f"🔒 {text}"
    if "Approval request" in m and "created" in m:
        return f"🔒 **Approval gate opened** — awaiting human decision"

    # ── Graph execution complete ──────────────────────────────────────────────
    if "Graph execution complete" in m:
        return "✅ **Analysis complete**"

    # ── Fallback: show as-is (indented, smaller) ─────────────────────────────
    return f"&emsp;{m}"


def call_analysis_api_with_streaming(user_input: str, log_placeholder, conversation_history: list = None) -> Optional[dict]:
    """
    Call the streaming backend endpoint and render agent logs LIVE as they arrive.

    Consumes newline-delimited JSON (NDJSON) events from /api/analyze/stream:
        {"type": "log", "message": ...}   -> appended to the live log view
        {"type": "result", "result": ...} -> the final AnalysisResponse payload

    Args:
        user_input: User query
        log_placeholder: Streamlit placeholder updated in real-time with logs
        conversation_history: Prior chat turns for follow-up context

    Returns:
        Final response JSON if successful, None otherwise
    """
    try:
        payload = {"user_input": user_input, "conversation_history": conversation_history or []}
        logs: list = []       # raw log lines (kept for metadata)
        steps: list = []      # cleaned display steps — FRESH for each call
        final_result = None
        
        # Clear placeholder to ensure no old content is displayed
        log_placeholder.empty()

        with requests.post(
            API_ANALYZE_STREAM_ENDPOINT,
            json=payload,
            timeout=REQUEST_TIMEOUT,
            stream=True,
        ) as response:
            if response.status_code != 200:
                st.error(f"❌ API Error: {response.status_code}")
                try:
                    st.write("Response:", response.json())
                except Exception:
                    st.write("Response:", response.text)
                return None

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")
                if event_type == "log":
                    raw = event.get("message", "")
                    logs.append(raw)
                    level, msg = _extract_log_message(raw)
                    if msg:
                        step = _format_agent_step(level, msg)
                        if step:
                            steps.append(step)
                            # Always show last 25 steps, ensuring placeholder is kept fresh
                            log_placeholder.markdown("\n\n".join(steps[-25:]))
                elif event_type == "result":
                    final_result = event.get("result")

        if final_result is not None:
            # Attach the full streamed log history for downstream display
            final_result.setdefault("logs", logs)
            # Persist raw sources so the sidebar citation panel can display them
            if final_result.get("sources"):
                st.session_state.last_sources = final_result["sources"]
                st.session_state.last_sources_ticker = (
                    final_result.get("routing_decision", "")
                )
            return final_result

        st.warning("⚠️ Stream ended without a final result.")
        return None

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ **Backend Connection Error**: Could not reach the FastAPI server. "
            "Please ensure the backend is running:\n\n"
            "```\nC:\\Users\\anand\\anaconda3\\python.exe backend.py\n```"
        )
        return None
    except requests.exceptions.Timeout:
        st.error(
            "❌ **Timeout Error**: The backend took longer than 10 minutes to respond. "
            "\n\nPossible causes:"
            "\n- Qwen LLM is taking too long (committee runs multiple calls)"
            "\n- Backend crashed or is stuck"
            "\n- Network connection lost"
            "\n\nCheck the backend terminal for errors."
        )
        return None
    except Exception as e:
        st.error(f"❌ **Error**: {str(e)}")
        import traceback
        st.write(traceback.format_exc())
        return None


def call_analysis_api(user_input: str, conversation_history: list = None) -> Optional[dict]:
    """
    Call the FastAPI backend to analyze the user input.

    Args:
        user_input: User query
        conversation_history: Prior chat turns for follow-up context

    Returns:
        Response JSON if successful, None otherwise

    Notes:
        Supports request tracking for live log polling.
        Current implementation waits for full response (blocking).

        Architecture:
        1. Creates request with ID on backend
        2. Logs are captured per-request in backend
        3. Frontend polls /api/requests/{request_id}/status for live logs
        4. Response includes request_id for future polling/cancellation
    """
    try:
        payload = {"user_input": user_input, "conversation_history": conversation_history or []}
        
        response = requests.post(
            API_ANALYZE_ENDPOINT,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error(
            "❌ **Backend Connection Error**: Could not reach the FastAPI server. "
            "Please ensure the backend is running with `uvicorn backend:app --reload`"
        )
        return None
    except requests.exceptions.Timeout:
        st.error(
            "❌ **Timeout Error**: The backend took too long to respond (5 minutes). "
            "\n\nPossible reasons:"
            "\n- Qwen LLM is generating a detailed response (can take 1-2 minutes)"
            "\n- Backend is stuck on a single operation"
            "\n- Network latency is high"
            "\n\nTip: Check the backend logs to see what operation is taking time."
        )
        return None
    except Exception as e:
        st.error(f"❌ **Error**: {str(e)}")
        return None


def get_routing_badge_class(routing_decision: str) -> str:
    """Get CSS class for routing decision badge."""
    if routing_decision in ("research", "completed", "awaiting_approval"):
        return "research-badge"
    elif routing_decision == "general_q":
        return "general-badge"
    else:
        return ""


def get_routing_badge_label(routing_decision: str) -> str:
    """Get display label for routing decision."""
    if routing_decision in ("research", "completed", "awaiting_approval"):
        return "🏛️ Investment Committee"
    elif routing_decision == "general_q":
        return "💡 General Question"
    else:
        return "❓ Unknown"


# ============================================================================
# PAGE HEADER
# ============================================================================

st.markdown(
    "<div class='header-container'><h1 class='main-title'>🧠 Corporate Intelligence & Earnings Analyst</h1></div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style='text-align: center; margin-bottom: 30px;'>
        <p style='color: #6b7280; font-size: 1.1em; margin: 5px 0; font-weight: 500;'>
            Powered by Multi-Agent AI Orchestration
        </p>
        <p style='color: #9ca3af; font-size: 0.95em; margin: 5px 0;'>
            Real-time Financial Analysis • SEC Data • Market Intelligence
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# SIDEBAR - STATUS & HELP
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.caption(f"🔗 Connected to: `{API_BASE_URL}`")
    
    # API Health Status
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Backend Status**")
    with col2:
        if check_api_health():
            st.markdown("<span style='color: #28a745; font-weight: bold;'>✓ Online</span>", unsafe_allow_html=True)
            st.session_state.api_available = True
        else:
            st.markdown("<span style='color: #dc3545; font-weight: bold;'>✗ Offline</span>", unsafe_allow_html=True)
            st.session_state.api_available = False
    
    if not st.session_state.api_available:
        st.warning(
            "⚠️ **Backend not running!**\n\n"
            "Start the FastAPI server in another terminal:\n\n"
            "```bash\n"
            "uvicorn backend:app --reload\n"
            "```"
        )

    st.divider()

    # ── Portfolio Panel ──────────────────────────────────────────────────
    st.markdown("<h3 style='font-size: 1.05em; font-weight: 600;'>📊 Portfolio</h3>", unsafe_allow_html=True)
    _refresh_key = "portfolio_refresh"
    if st.button("🔄 Refresh Portfolio", key=_refresh_key, use_container_width=True):
        st.session_state.pop("portfolio_data", None)

    if st.session_state.get("api_available"):
        if "portfolio_data" not in st.session_state:
            try:
                _resp = requests.get(API_PORTFOLIO_ENDPOINT, timeout=30)
                if _resp.status_code == 200:
                    st.session_state.portfolio_data = _resp.json()
                else:
                    st.session_state.portfolio_data = None
            except Exception:
                st.session_state.portfolio_data = None

        _pf = st.session_state.get("portfolio_data")
        if _pf:
            _pnl = _pf.get("total_pnl", 0)
            _pnl_pct = _pf.get("total_pnl_pct", 0)

            # Enhanced metrics display
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                    <div class='metric-box'>
                        <div class='metric-label'>Total Value</div>
                        <div class='metric-value'>$""" + f"{_pf.get('total_value', 0):,.2f}" + """</div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("""
                    <div class='metric-box'>
                        <div class='metric-label'>Cash Balance</div>
                        <div class='metric-value'>$""" + f"{_pf.get('cash', 0):,.2f}" + """</div>
                    </div>
                """, unsafe_allow_html=True)

            pnl_color = "#10b981" if _pnl >= 0 else "#ef4444"
            pnl_symbol = "📈" if _pnl >= 0 else "📉"
            st.markdown(f"""
                <div class='metric-box' style='border-left: 5px solid {pnl_color}; padding: 12px;'>
                    <div class='metric-label'>Unrealised P&L</div>
                    <div class='metric-value' style='color: {pnl_color}; font-size: 1.3em; margin: 4px 0 2px 0;'>
                        {pnl_symbol} ${_pnl:+,.2f}
                    </div>
                    <div style='font-size: 0.75em; color: #6b7280;'>{_pnl_pct:+.2f}%</div>
                </div>
            """, unsafe_allow_html=True)

            holdings = _pf.get("holdings", [])
            if holdings:
                st.markdown("**📍 Holdings**")
                rows = ""
                for h in holdings:
                    pnl = h.get("unrealised_pnl", 0)
                    pnl_pct = h.get("unrealised_pct", 0)
                    pnl_color = "#10b981" if pnl >= 0 else "#ef4444"
                    pnl_icon = "▲" if pnl >= 0 else "▼"
                    rows += (
                        f"<tr>"
                        f"<td><b>{h['ticker']}</b></td>"
                        f"<td>${h.get('live_price', 0):,.2f}</td>"
                        f"<td>${h.get('market_value', 0):,.2f}</td>"
                        f"<td>{h.get('allocation_pct', 0)}%</td>"
                        f"<td style='color:{pnl_color}'>{pnl_icon} {pnl:+.2f}<br>"
                        f"<small style='opacity:0.8'>({pnl_pct:+.1f}%)</small></td>"
                        f"</tr>"
                    )
                st.markdown(
                    f"<table style='width:100%;font-size:0.82rem;border-collapse:collapse'>"
                    f"<thead><tr style='font-size:0.75rem;color:#555'>"
                    f"<th style='text-align:left'>Ticker</th>"
                    f"<th style='text-align:left'>Price</th>"
                    f"<th style='text-align:left'>Value</th>"
                    f"<th style='text-align:left'>Alloc</th>"
                    f"<th style='text-align:left'>P&amp;L</th>"
                    f"</tr></thead><tbody>{rows}</tbody></table>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No open positions")

            if st.button("🤖 Deep Analysis", key="portfolio_deep", use_container_width=True):
                # Pre-fill the chat with a portfolio analysis request
                st.session_state["_portfolio_query"] = "Analyze my portfolio"
                st.rerun()
        else:
            st.caption("Could not load portfolio data")
    else:
        st.caption("Backend offline")

    st.divider()

    # ── Citation / Sources Panel ─────────────────────────────────────────
    st.markdown("### 📚 Sources")
    _sources = st.session_state.get("last_sources")
    if _sources:
        st.caption("Raw data fed to the AI — click to expand")

        # [1] Yahoo Finance MCP Live Quote
        av_q = _sources.get("av_quote") or {}
        if av_q:
            with st.expander(f"**[1]** Yahoo Finance MCP — Live Quote", expanded=False):
                st.markdown(f"**Ticker:** {av_q.get('ticker', 'N/A')}")
                st.markdown(f"**Price:** ${av_q.get('price', 'N/A')}")
                st.markdown(f"**Change:** {av_q.get('change', 'N/A')} ({av_q.get('change_percent', 'N/A')}%)")
                st.markdown(f"**Volume:** {av_q.get('volume', 'N/A'):,}" if isinstance(av_q.get('volume'), int) else f"**Volume:** {av_q.get('volume', 'N/A')}")
                st.markdown(f"**Trading Day:** {av_q.get('timestamp', 'N/A')}")
                st.markdown(f"**Fetched at:** {av_q.get('fetched_at', 'N/A')}")
                st.caption(f"Source: {av_q.get('data_source', 'Yahoo Finance MCP')}")

        # [2] Yahoo Finance MCP Company Overview
        av_f = _sources.get("av_fundamentals") or {}
        if av_f:
            with st.expander("**[2]** Yahoo Finance MCP — Company Overview", expanded=False):
                _fund_labels = {
                    "sector": "Sector", "industry": "Industry",
                    "market_cap": "Market Cap", "pe_ratio": "P/E Ratio",
                    "forward_pe": "Forward P/E", "eps": "EPS (TTM)",
                    "analyst_target_price": "Analyst Target",
                    "week_52_high": "52-Week High", "week_52_low": "52-Week Low",
                    "profit_margin": "Profit Margin",
                    "quarterly_earnings_growth_yoy": "Earnings Growth YoY",
                    "quarterly_revenue_growth_yoy": "Revenue Growth YoY",
                    "beta": "Beta", "return_on_equity": "Return on Equity",
                    "dividend_yield": "Dividend Yield",
                }
                for key, label in _fund_labels.items():
                    val = av_f.get(key)
                    if val:
                        prefix = "$" if key in ("market_cap", "analyst_target_price", "eps", "week_52_high", "week_52_low") else ""
                        st.markdown(f"**{label}:** {prefix}{val}")
                st.caption("Source: Yahoo Finance MCP")

        # [3+] SEC 10-Q Filings
        sec = _sources.get("sec_filings") or {}
        if sec.get("available") and sec.get("quarters"):
            _sec_labels = {
                "revenue": "Revenue", "net_income": "Net Income",
                "gross_profit": "Gross Profit", "operating_income": "Operating Income",
                "rd_expense": "R&D Expense", "free_cash_flow": "Free Cash Flow",
                "cash": "Cash & Equivalents", "long_term_debt": "Long-term Debt",
                "total_assets": "Total Assets", "shareholders_equity": "Shareholders' Equity",
                "operating_cash_flow": "Operating Cash Flow",
                "eps_diluted": "EPS (Diluted)",
            }
            from app.tools.sec_tools import fmt_sec_value
            for i, q in enumerate(sec["quarters"]):
                cite_num = 3 + i
                with st.expander(
                    f"**[{cite_num}]** SEC 10-Q — Q-{i+1} "
                    f"(period ending {q.get('period', 'N/A')})",
                    expanded=False,
                ):
                    st.markdown(f"**Period end:** {q.get('period', 'N/A')}")
                    st.markdown(f"**Filed:** {q.get('filed', 'N/A')}")
                    st.markdown("---")
                    for key, label in _sec_labels.items():
                        val = q.get(key)
                        if val is None:
                            continue
                        if key == "eps_diluted":
                            st.markdown(f"**{label}:** ${val:.2f}")
                        else:
                            st.markdown(f"**{label}:** {fmt_sec_value(val)}")
                    st.caption("Source: SEC EDGAR via edgartools (XBRL)")
    else:
        st.caption("No sources loaded yet. Run an analysis query to see citation data here.")

    st.divider()

    # ── Help ─────────────────────────────────────────────────────────────
    st.markdown("### 📖 Help & Examples")
    st.markdown("""
    **What can I ask?**
    *   **📊 Research:** `"Analyze NVDA"` | `"Compare MSFT vs GOOG"` | `"Should I buy TSLA?"`
    *   **💰 Trading:** `"Buy $500 of NVDA"` | `"Sell $100 shares of Tesla stock"`
    *   **💼 Portfolio:** `"Show my portfolio"` | `"What's my concentration?"`
    *   **📚 Education:** `"What is market cap?"` | `"Explain P/E ratio"`
    """)
    
    st.divider()
    
    st.markdown("### 🎯 AI Agent Skills")
    st.markdown("""
    - **Portfolio Analyzer**: Evaluates asset allocation, diversification, and optimization. *(Try: "Analyze my positions")*
    - **Risk Manager**: Assesses volatility, correlation, and liquidity risk scores (0-100). *(Try: "What's my portfolio risk?")*
    - **Smart Triggers**: Intercepts trades breaching stop-loss (-20%), take-profit (+50%), or allocation limits (50%). *(Try: "Should I buy more TSLA?")*
    """)
    
    st.divider()
    
    st.markdown("### 💬 Context-Aware Follow-ups")
    st.markdown("""
    The engine retains conversational context, resolving ambiguous follow-ups automatically:
    
    *   **Flow 1:** `"Analyze TSLA"` ➔ `"How does profitability compare to competitors?"`
    *   **Flow 2:** `"Analyze AAPL"` ➔ `"Buy $1000 of the above stock"`
    """)
    
    st.divider()

    st.markdown("### ⚠️ Disclaimer")
    st.info(
        "**Disclaimer**: This information is for educational purposes only and does not "
        "constitute financial advice. Always conduct your own research or consult with a "
        "qualified financial advisor before making investment decisions."
    )


# ============================================================================
# MAIN CONTENT - CHAT INTERFACE
# ============================================================================

st.markdown("""
<div class='interface-box'>
    <div class='interface-title'>💬 Analysis Interface</div>
    <div class='interface-subtitle'>Ask questions about stocks, market analysis, or your portfolio</div>
</div>
""", unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Fix concatenated text, bullet formatting, and escape currency for all content
        # EXCEPT trade execution messages which are already properly formatted
        if message["role"] == "assistant" and message.get("metadata", {}).get("type") != "trade_execution":
            content = _format_bullet_points(_fix_concatenated_text(message["content"]))
        else:
            content = message["content"]
        st.markdown(_escape_currency(content))
        
        # Display metadata if it's an assistant response
        if message["role"] == "assistant" and "metadata" in message:
            metadata = message["metadata"]
            is_trade_execution = metadata.get("type") == "trade_execution"
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if is_trade_execution:
                    trade_info = metadata.get("trade_info", {})
                    trade_status = trade_info.get("status", "N/A")
                    st.markdown(f"""
                        <div style='font-size: 0.75em; color: #6b7280;'>
                            <strong>Trade Status</strong><br>
                            <span style='font-size: 0.9em; color: #1f2937;'>✅ {trade_status}</span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style='font-size: 0.75em; color: #6b7280;'>
                            <strong>Routing Decision</strong><br>
                            <span style='font-size: 0.9em; color: #1f2937;'>{get_routing_badge_label(metadata.get("routing_decision", "Unknown"))}</span>
                        </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div style='font-size: 0.75em; color: #6b7280;'>
                        <strong>Execution Time</strong><br>
                        <span style='font-size: 0.9em; color: #1f2937;'>{metadata.get('execution_time_ms', 0):.1f}ms</span>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div style='font-size: 0.75em; color: #6b7280;'>
                        <strong>Log Entries</strong><br>
                        <span style='font-size: 0.9em; color: #1f2937;'>{len(metadata.get("logs", []))}</span>
                    </div>
                """, unsafe_allow_html=True)
            
            # ── Render archived logs from history ──
            past_logs = metadata.get("logs", [])
            if past_logs:
                with st.expander("👁️ View Agent Execution Logs", expanded=False):
                    formatted_steps = []
                    for raw_line in past_logs:
                        level, msg = _extract_log_message(raw_line)
                        if msg:
                            step = _format_agent_step(level, msg)
                            if step:
                                formatted_steps.append(step)
                    
                    if formatted_steps:
                        st.markdown("\n\n".join(formatted_steps))
                    else:
                        st.caption("No displayable agent steps found in logs.")
            
            # ── Render trade notification only within its own history turn ──
            trade_info = metadata.get("trade_info")
            if trade_info:
                with st.expander(f"📊 **Trade:** {trade_info['trade_details'].get('action', 'N/A')} {trade_info['trade_details'].get('ticker', 'N/A')} ({trade_info['status']})", expanded=False):
                    if trade_info["status"] == "EXECUTED":
                        execution = trade_info.get("execution_result", {})
                        st.success("✅ Trade executed successfully")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Filled Price", f"${execution.get('filled_price', 0):.2f}")
                        with col2:
                            st.metric("Quantity", f"{execution.get('quantity', 0):.4f} shares")
                        st.caption(f"Order ID: `{execution.get('order_id', 'N/A')}`")
                    else:
                        st.error("❌ Trade rejected")
                        st.caption(f"Request ID: `{trade_info.get('request_id', 'N/A')}`")


# ============================================================================
# USER INPUT & PROCESSING
# ============================================================================

# Chat input box
# Chat input box — also accepts pre-fills from sidebar "Deep Analysis" button
user_input = st.chat_input(
    "🔍 Ask about stocks, market trends, or analyze your portfolio...",
    key="user_input",
)

# Allow sidebar "Deep Analysis" button to inject a query
if not user_input and st.session_state.pop("_portfolio_query", None):
    user_input = "Analyze my portfolio"

if user_input:
    # Reset trade completion flag for new requests
    st.session_state.trade_completed = False
    
    # IMPORTANT: Clear any cached log state to prevent old responses from preloading
    # This ensures each new query starts with a clean slate
    st.session_state.pop("_last_analysis_logs", None)
    st.session_state.pop("_last_streaming_steps", None)
    
    # IMPORTANT: Clear previous query's sources so sidebar doesn't show stale citations
    # Sources will be repopulated with current query's results as soon as it finishes
    st.session_state.pop("last_sources", None)
    st.session_state.pop("last_sources_ticker", None)
    
    # Check API availability first
    if not check_api_health():
        st.error(
            "❌ **Cannot process request**: Backend is not running. "
            "Please start the FastAPI server first."
        )
    else:
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Create status container for real-time updates
        with st.status("🚀 Orchestrating AI Agents...", expanded=True) as status:
            st.write("Initiating multi-agent analysis workflow...")
            
            # Create a FRESH placeholder for this query's logs
            # The empty() ensures no cached content from previous queries
            log_placeholder = st.empty()
            log_placeholder.markdown("")  # Explicitly render empty to clear cache
            
            # Build conversation history - Keep last 5 turns for richer context
            # This preserves ticker and analysis context across multiple follow-up questions
            history_turns = []
            
            # Include last 5 turns (user + assistant pairs)
            for m in st.session_state.messages[:-1]:  # exclude current user msg
                # Keep full content + metadata for proper ticker disambiguation
                turn = {"role": m["role"], "content": m["content"]}
                
                # For assistant messages, include routing decision and ticker context
                if m["role"] == "assistant" and "metadata" in m:
                    metadata = m.get("metadata", {})
                    # Add routing info to help backend understand context
                    if metadata.get("routing_decision"):
                        turn["routing_decision"] = metadata["routing_decision"]
                
                history_turns.append(turn)
            
            # Only keep the last ~10 messages to balance context vs token usage (5 turns)
            history_turns = history_turns[-10:]

            # Call the API with streaming logs
            response = call_analysis_api_with_streaming(user_input, log_placeholder, conversation_history=history_turns)
            
            if response:
                # Check for errors
                if response.get("status") == "error":
                    status.update(
                        label="❌ Analysis Failed!",
                        state="error",
                        expanded=False,
                    )
                    st.error(f"Error: {response.get('error_message', 'Unknown error')}")
                
                # Check for approval pending
                elif response.get("status") == "awaiting_approval":
                    status.update(
                        label="⏳ Awaiting Human Approval",
                        state="running",
                        expanded=True,
                    )
                    
                    # Store approval request in session state so it persists after container closes
                    pending_approval = response.get("pending_approval", {})
                    request_id = response.get("request_id", "")
                    st.session_state.pending_approval_request = {
                        "request_id": request_id,
                        "pending_approval": pending_approval,
                        "response": response,
                    }
                    
                    st.warning("🔒 **TRADE APPROVAL REQUIRED** - Review below")
                    # All approval UI is handled in the persistent section below
                
                else:
                    # Close the status container — report renders below it
                    status.update(
                        label=f"✅ Analysis Complete! ({response.get('execution_time_ms', 0):.1f}ms)",
                        state="complete",
                        expanded=False,
                    )
            else:
                status.update(
                    label="❌ Request Failed",
                    state="error",
                    expanded=False,
                )

        # ── Report rendering (OUTSIDE status container so it's never collapsed) ──
        if response and response.get("status") not in ("error", "awaiting_approval"):
            report_md = response.get("report_markdown", "")

            # Save to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": report_md,
                "metadata": {
                    "routing_decision": response.get("routing_decision", ""),
                    "logs": response.get("logs", []),
                    "execution_time_ms": response.get("execution_time_ms", 0),
                }
            })

            # Render report in a chat bubble
            with st.chat_message("assistant"):
                st.markdown(
                    f"<div class='routing-badge {get_routing_badge_class(response.get('routing_decision'))}'>"
                    f"{get_routing_badge_label(response.get('routing_decision'))}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                # Fix concatenated text, bullet formatting, and escape currency
                cleaned_report = _format_bullet_points(_fix_concatenated_text(report_md))
                escaped_report = _escape_currency(cleaned_report)
                
                # Render report directly (no streaming) to avoid partial markdown rendering issues
                st.markdown(escaped_report)
            
            # Force immediate rerun to refresh sidebar with new sources
            st.rerun()


# =========================================================================
# APPROVAL UI - OUTSIDE THE STATUS CONTAINER (PERSISTS AFTER CONTAINER CLOSES)
# =========================================================================

# Only show approval UI if trade hasn't been completed
if st.session_state.get("pending_approval_request") and not st.session_state.get("trade_completed"):
    approval_data = st.session_state.pending_approval_request
    request_id = approval_data.get("request_id", "")
    pending_approval = approval_data.get("pending_approval", {})
    
    st.markdown("---")
    st.warning("🔒 **TRADE APPROVAL REQUIRED** - Click buttons below")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Trade Details:**")
        st.info(
            f"**Action:** {pending_approval.get('action', 'N/A')}\n\n"
            f"**Ticker:** {pending_approval.get('ticker', 'N/A')}\n\n"
            f"**Amount:** {pending_approval.get('amount', 'N/A')}\n\n"
            f"**Request ID:** `{request_id}`"
        )
    
    with col2:
        st.markdown("**Reasoning:**")
        st.text(
            pending_approval.get('reasoning', 'No reasoning provided')
        )
    
    # Approval buttons
    st.markdown("---")
    st.markdown("### ✅ Approval Decision")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Approve Trade", key=f"approve_{request_id}_persistent", use_container_width=True):
            with st.spinner("Submitting approval and executing trade..."):
                try:
                    # Step 1: Submit approval
                    approve_response = requests.post(
                        f"{API_BASE_URL}/api/approve/{request_id}",
                        json={"approved": True, "approver_notes": "Approved via Streamlit UI"},
                        timeout=120  # Increased for OAuth/MCP
                    )
                    
                    if approve_response.status_code != 200:
                        error_resp = approve_response.json()
                        st.error(f"❌ Failed to submit approval: {approve_response.status_code}")
                        st.error(
                            f"**Error:** {error_resp.get('detail', 'Unknown error')}\n\n"
                            f"**Status Code:** {approve_response.status_code}"
                        )
                    else:
                        st.success("✅ Approval stored!")
                        
                        # Step 2: Execute the trade
                        time.sleep(0.5)
                        execute_response = requests.post(
                            f"{API_BASE_URL}/api/execute/{request_id}",
                            timeout=120  # Increased for OAuth/MCP
                        )
                        
                        if execute_response.status_code == 200:
                            result = execute_response.json()
                            st.success("✅ **TRADE EXECUTED SUCCESSFULLY!**")
                            
                            # Format and display the trade execution result
                            execution_result = result.get("execution_result", {})
                            
                            st.markdown("### 📊 Order Execution")
                            # Handle None values explicitly
                            try:
                                filled_price = float(execution_result.get('filled_price') or 0)
                            except (ValueError, TypeError):
                                filled_price = 0.0
                            
                            try:
                                quantity = float(execution_result.get('quantity') or 0)
                            except (ValueError, TypeError):
                                quantity = 0.0
                            
                            st.success(
                                f"**Order ID:** `{execution_result.get('order_id', 'N/A')}`\n\n"
                                f"**Status:** {execution_result.get('status', 'N/A').upper()}\n\n"
                                f"**Fill Price:** ${filled_price:.2f}\n\n"
                                f"**Quantity Filled:** {quantity:.4f} shares"
                            )
                            
                            st.markdown("---")
                            st.markdown("### ✅ Summary")
                            st.success(
                                f"💰 **Total Value:** ${execution_result.get('total_value', 0):.2f}\n\n"
                                f"📅 **Filled At:** {execution_result.get('filled_at', 'N/A')}\n\n"
                                f"🎮 **Simulated Trade:** {'Yes' if execution_result.get('simulated') else 'No'}"
                            )
                            
                            # Store trade result for persistence
                            trade_info = {
                                "status": "EXECUTED",
                                "trade_details": result.get("trade_details", {}),
                                "execution_result": execution_result
                            }
                            st.session_state.last_completed_trade = trade_info
                            
                            # Create a new message for the trade execution result
                            trade_execution_message = {
                                "role": "assistant",
                                "content": (
                                    f"✅ **TRADE EXECUTED SUCCESSFULLY!**\n\n"
                                    f"**Order ID:** `{execution_result.get('order_id', 'N/A')}`\n\n"
                                    f"**Status:** {execution_result.get('status', 'N/A').upper()}\n\n"
                                    f"**Fill Price:** ${filled_price:.2f}\n\n"
                                    f"**Quantity Filled:** {quantity:.4f} shares\n\n"
                                    f"**Total Value:** ${execution_result.get('total_value', 0):.2f}\n\n"
                                    f"**Filled At:** {execution_result.get('filled_at', 'N/A')}"
                                ),
                                "metadata": {
                                    "type": "trade_execution",
                                    "trade_info": trade_info,
                                    "logs": result.get("logs", []),
                                    "execution_time_ms": result.get("execution_time_ms", 0)
                                }
                            }
                            
                            # Add the trade execution message to the chat history
                            st.session_state.messages.append(trade_execution_message)
                            
                            # Clear the approval request and refresh portfolio
                            st.session_state.pending_approval_request = None
                            st.session_state.trade_completed = True
                            st.session_state.pop("portfolio_data", None)  # refresh sidebar
                            st.rerun()  # Auto-refresh page to display updated portfolio
                        else:
                            error_resp = execute_response.json()
                            st.error(f"❌ Execution failed: {execute_response.status_code}")
                            st.error(
                                f"**Error:** {error_resp.get('detail', 'Unknown error')}\n\n"
                                f"**Status Code:** {execute_response.status_code}"
                            )
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.write(traceback.format_exc())
    
    with col2:
        if st.button("❌ Reject Trade", key=f"reject_{request_id}_persistent", use_container_width=True):
            with st.spinner("Submitting rejection..."):
                try:
                    reject_response = requests.post(
                        f"{API_BASE_URL}/api/approve/{request_id}",
                        json={"approved": False, "approver_notes": "Rejected via Streamlit UI"},
                        timeout=120  # Increased for OAuth/MCP
                    )
                    
                    if reject_response.status_code == 200:
                        st.error("❌ **TRADE REJECTED**")
                        
                        # Store rejection for persistence
                        trade_info = {
                            "status": "REJECTED",
                            "trade_details": approval_data.get("pending_approval", {}),
                            "request_id": request_id
                        }
                        st.session_state.last_completed_trade = trade_info
                        
                        # Also attach to last message's metadata so it displays with that turn
                        if st.session_state.messages:
                            msg = st.session_state.messages[-1]
                            msg.setdefault("metadata", {})["trade_info"] = trade_info
                        
                        # Clear the approval request and refresh portfolio
                        st.session_state.pending_approval_request = None
                        st.session_state.trade_completed = True
                        st.session_state.pop("portfolio_data", None)  # refresh sidebar
                        st.rerun()  # Auto-refresh page to display updated portfolio
                    else:
                        error_resp = reject_response.json()
                        st.error(f"Failed to submit rejection: {reject_response.status_code}")
                        st.error(
                            f"**Error:** {error_resp.get('detail', 'Unknown error')}\n\n"
                            f"**Status Code:** {reject_response.status_code}"
                        )
                except Exception as e:
                    st.error(f"Error: {str(e)}")


# ============================================================================
# FOOTER & INFO
# ============================================================================

st.divider()

# About / Technical Stack (Collapsible)
with st.expander("ℹ️ About This Platform"):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.info("💻 **Frontend**: Streamlit")
    
    with col2:
        st.info("⚙️ **Backend**: FastAPI on Alibaba Cloud FC")
    
    with col3:
        st.info("🧠 **Orchestration**: LangGraph State Machine")
    
    with col4:
        st.info("🤖 **LLM**: Qwen (Alibaba)")

st.markdown("""
---
<p style='text-align: center; color: #999; font-size: 0.9em;'>
Corporate Intelligence Engine | AI State Graph | Financial Research
</p>
""", unsafe_allow_html=True)
