import json
import re
import os
import io
import html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pypdf import PdfReader

# Google GenAI SDK
from google import genai
from google.genai import types

# LangChain & FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ReportLab Libraries for PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================
GEMINI_MODEL = "gemini-3.1-flash-lite"

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION (SIDEBAR FULLY COLLAPSED / DISABLED)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="IntellectPDF | Multi-Agent Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------------------------
# ENTERPRISE LIGHT-THEME CSS (SIDEBAR-FREE, GLASSMORPHIC, INTER TYPOGRAPHY)
# ------------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* ---- Completely remove the sidebar ---- */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }

    /* ---- Main App Light Background & Typography ---- */
    .stApp {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #0F172A !important;
    }
    .block-container {
        max-width: 1250px !important;
        padding-top: 4rem !important;
        padding-bottom: 3rem !important;
    }

    /* ---- Top Brand Bar ---- */
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 0 20px 0;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 24px;
    }
    .topbar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .topbar-logo {
        font-size: 1.9rem;
        background: linear-gradient(135deg, #4F46E5, #2563EB);
        width: 46px; height: 46px;
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 10px rgba(79, 70, 229, 0.28);
    }
    .topbar-title { font-size: 1.35rem; font-weight: 800; color: #0F172A !important; line-height: 1.1; }
    .topbar-caption { font-size: 0.78rem; color: #64748B !important; font-weight: 500; }

    /* ---- Hero Section ---- */
    .hero-wrap { text-align: center; padding: 30px 10px 10px 10px; }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 9px;
        background: #EEF2FF; border: 1px solid #C7D2FE; color: #4338CA !important;
        padding: 7px 18px; border-radius: 999px; font-size: 0.8rem; font-weight: 700;
        margin-bottom: 26px; letter-spacing: 0.02em;
    }
    .hero-badge .dot {
        width: 8px; height: 8px; border-radius: 50%; background: #4F46E5;
        animation: pulseDot 1.7s infinite;
    }
    @keyframes pulseDot {
        0% { box-shadow: 0 0 0 0 rgba(79,70,229,0.55); }
        70% { box-shadow: 0 0 0 9px rgba(79,70,229,0); }
        100% { box-shadow: 0 0 0 0 rgba(79,70,229,0); }
    }
    .hero-title {
        font-size: 3.1rem; font-weight: 900; line-height: 1.14; margin-bottom: 18px;
        color: #0F172A !important; letter-spacing: -0.02em;
    }
    .hero-title .gradient {
        background: linear-gradient(90deg, #4F46E5, #7C3AED 55%, #2563EB);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-subtitle {
        font-size: 1.14rem; color: #475569 !important; max-width: 700px;
        margin: 0 auto 8px auto; line-height: 1.65; font-weight: 450;
    }

    /* ---- Upload Card ---- */
    /* ---- Animated Glowing & Moving Border Upload Card ---- */
    .upload-label {
        text-align: center; font-weight: 750; font-size: 1.05rem; color: #0F172A !important;
        margin-bottom: 6px; margin-top: 10px;
    }
    .upload-sublabel {
        text-align: center; font-size: 0.86rem; color: #64748B !important; margin-bottom: 16px;
    }

    [data-testid="stFileUploaderDropzone"] {
        position: relative !important;
        background: #FFFFFF !important;
        border-radius: 18px !important;
        border: 2px solid transparent !important;
        background-image: linear-gradient(#FFFFFF, #FFFFFF), 
                          linear-gradient(90deg, #4F46E5, #06B6D4, #7C3AED, #4F46E5) !important;
        background-origin: border-box !important;
        background-clip: padding-box, border-box !important;
        background-size: 300% 300% !important;
        animation: borderMove 4s linear infinite, glowPulse 2.5s ease-in-out infinite alternate !important;
        box-shadow: 0 4px 20px rgba(79, 70, 229, 0.12) !important;
        transition: all 0.3s ease !important;
        padding: 20px !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        transform: translateY(-3px) scale(1.01) !important;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25) !important;
    }

    /* Animation Keyframes: Border Move & Glow */
    @keyframes borderMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes glowPulse {
        0% {
            box-shadow: 0 0 12px rgba(79, 70, 229, 0.2), inset 0 0 8px rgba(79, 70, 229, 0.04);
        }
        100% {
            box-shadow: 0 0 26px rgba(124, 58, 237, 0.45), inset 0 0 12px rgba(6, 182, 212, 0.1);
        }
    }

    /* ---- Feature Showcase Grid ---- */
    .feature-grid {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 44px;
    }
    @media (max-width: 900px) { .feature-grid { grid-template-columns: 1fr; } }
    .feature-card {
        background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px; padding: 26px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 14px 28px rgba(79, 70, 229, 0.14);
        border-color: #C7D2FE;
    }
    .feature-icon {
        font-size: 1.7rem; margin-bottom: 14px; width: 48px; height: 48px;
        background: #EEF2FF; border-radius: 12px; display: flex; align-items: center; justify-content: center;
    }
    .feature-title { font-weight: 750; font-size: 1.03rem; margin-bottom: 7px; color: #0F172A !important; }
    .feature-desc { font-size: 0.87rem; color: #64748B !important; line-height: 1.55; }

    /* ---- Section Card Header (inside bordered containers) ---- */
    .section-header {
        display: flex; align-items: center; gap: 14px;
        padding: 4px 4px 16px 4px; margin-bottom: 12px; border-bottom: 1px solid #F1F5F9;
    }
    .section-icon {
        font-size: 1.5rem; background: #EEF2FF; color: #4F46E5;
        width: 46px; height: 46px; border-radius: 13px;
        display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    }
    .section-title { font-size: 1.28rem; font-weight: 800; color: #0F172A !important; margin: 0; }
    .section-subtitle { font-size: 0.86rem; color: #64748B !important; margin: 2px 0 0 0; }

    /* ---- Bordered Section "Card" Containers ---- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 18px !important;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.045);
        margin-bottom: 26px;
    }

    /* ---- Tabs Styling ---- */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #F1F5F9 !important;
        border-radius: 10px;
        padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #475569 !important;
        font-weight: 600;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #2563EB !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }

    /* ---- Buttons ---- */
    .stButton>button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 9px !important;
        font-weight: 650 !important;
        padding: 0.55rem 1.1rem !important;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #1D4ED8 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    }
    .stButton>button:disabled {
        background-color: #CBD5E1 !important; color: #64748B !important;
    }
    .stDownloadButton>button {
        background-color: #059669 !important;
        color: #FFFFFF !important;
        border-radius: 9px !important;
        font-weight: 650 !important;
    }
    .stDownloadButton>button:hover { background-color: #047857 !important; }

    /* ---- Reset Button Variant ---- */
    .reset-btn button {
        background-color: #FFFFFF !important;
        color: #DC2626 !important;
        border: 1px solid #FECACA !important;
        font-weight: 650 !important;
    }
    .reset-btn button:hover {
        background-color: #FEF2F2 !important;
        box-shadow: none !important;
    }

    /* ---- KPI Cards ---- */
    .kpi-card {
        background: #FFFFFF !important;
        border-left: 5px solid #2563EB !important;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: left;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        height: 100%;
    }
    .kpi-title {
        font-size: 0.78rem; text-transform: uppercase; color: #64748B !important;
        font-weight: 700; letter-spacing: 0.05em; margin-bottom: 5px;
    }
    .kpi-value { font-size: 1.45rem; font-weight: 800; color: #0F172A !important; }

    /* ---- Insight / Risk Badge Cards ---- */
    .insight-card {
        padding: 14px; border-radius: 10px; margin-bottom: 12px;
        font-size: 0.93rem; border-left: 5px solid;
    }
    .insight-positive { background: #ECFDF5 !important; border-color: #10B981 !important; color: #065F46 !important; }
    .insight-attention { background: #FFFBEB !important; border-color: #F59E0B !important; color: #92400E !important; }
    .insight-critical { background: #FEF2F2 !important; border-color: #EF4444 !important; color: #991B1B !important; }

    /* ---- Warning / Config Card ---- */
    .config-warning-card {
        max-width: 560px; margin: 60px auto; background: #FFFFFF;
        border: 1px solid #FEE2E2; border-radius: 16px; padding: 34px;
        text-align: center; box-shadow: 0 4px 16px rgba(220,38,38,0.06);
    }

    .stRadio label, .stTextInput label, .stSelectbox label, .stSlider label {
        color: #1E293B !important; font-weight: 500;
    }

    /* ---- Modern Enterprise Footer ---- */
    .app-footer {
        margin-top: 60px;
        padding-top: 25px;
        padding-bottom: 20px;
        border-top: 1px solid #E2E8F0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 15px;
    }
    .footer-left {
        font-size: 0.9rem;
        color: #64748B;
        font-weight: 500;
    }
    .footer-left strong {
        color: #0F172A;
    }
    .footer-socials {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .social-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 9px;
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        color: #475569;
        text-decoration: none !important;
        transition: all 0.2s ease-in-out;
    }
    .social-btn:hover {
        background: #EEF2FF;
        color: #2563EB;
        border-color: #C7D2FE;
        transform: translateY(-2px);
    }
    .social-btn svg {
        width: 18px;
        height: 18px;
        fill: currentColor;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ------------------------------------------------------------------------------
_DEFAULTS = {
    "vector_store": None,
    "doc_stats": {},
    "analytics_data": None,
    "quiz_data": None,
    "chat_history": [],
    "agent_outputs": {},
    "file_list": [],
    "corpus_sample": "",
    "uploader_version": 0,
    "cfg_chunk_size": 1000,
    "cfg_chunk_overlap": 200,
    "cfg_top_k": 4,
}
for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


def reset_corpus():
    """Clears all indexed-document state and returns the app to the landing page."""
    st.session_state.vector_store = None
    st.session_state.doc_stats = {}
    st.session_state.analytics_data = None
    st.session_state.quiz_data = None
    st.session_state.chat_history = []
    st.session_state.agent_outputs = {}
    st.session_state.file_list = []
    st.session_state.corpus_sample = ""
    st.session_state.uploader_version += 1


# ------------------------------------------------------------------------------
# HELPER FUNCTIONS & GEMINI CLIENT SETUP
# ------------------------------------------------------------------------------
def get_gemini_client(api_key: str):
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def clean_json_response(raw_response: str) -> str:
    raw_response = raw_response.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_response


@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def resolve_api_key() -> str:
    """Resolves the Gemini API key strictly from backend secrets / environment.
    No front-end input field is ever rendered for the key."""
    key = ""
    try:
        key = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        key = ""
    if not key:
        key = os.environ.get("GOOGLE_API_KEY", "")
    return key


# ------------------------------------------------------------------------------
# REPORTLAB PDF GENERATOR FUNCTION
# ------------------------------------------------------------------------------
def generate_pdf_report(doc_summary_name, stats, analytics_data, agent_outputs):
    """Generates an executive report PDF dynamically using ReportLab Platypus."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    story = []
    styles = getSampleStyleSheet()

    PRIMARY_COLOR = colors.HexColor("#1E293B")
    ACCENT_COLOR = colors.HexColor("#2563EB")
    TEXT_COLOR = colors.HexColor("#334155")
    BG_LIGHT = colors.HexColor("#F8FAFC")

    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold',
        fontSize=20, leading=24, textColor=PRIMARY_COLOR, spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle', parent=styles['Normal'], fontName='Helvetica-Oblique',
        fontSize=10, leading=14, textColor=colors.HexColor("#64748B"), spaceAfter=15
    )
    h2_style = ParagraphStyle(
        'SectionHeading', parent=styles['Heading2'], fontName='Helvetica-Bold',
        fontSize=13, leading=16, textColor=ACCENT_COLOR, spaceBefore=12, spaceAfter=8
    )
    body_style = ParagraphStyle(
        'BodyDark', parent=styles['Normal'], fontName='Helvetica',
        fontSize=9, leading=13, textColor=TEXT_COLOR, spaceAfter=6
    )

    # 1. Header & Metadata
    story.append(Paragraph("IntellectPDF: Executive Intelligence Report", title_style))
    story.append(Paragraph("IIT Patna Generative AI Capstone Project", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT_COLOR, spaceAfter=15))

    metadata_data = [
        [
            Paragraph("<b>Corpus Target:</b>", body_style),
            Paragraph(str(doc_summary_name), body_style),
            Paragraph("<b>Total Pages:</b>", body_style),
            Paragraph(str(stats.get('total_pages', 'N/A')), body_style)
        ],
        [
            Paragraph("<b>Est. Word Count:</b>", body_style),
            Paragraph(f"{stats.get('total_tokens', 0):,}", body_style),
            Paragraph("<b>FAISS Vector Nodes:</b>", body_style),
            Paragraph(str(stats.get('total_chunks', 'N/A')), body_style)
        ]
    ]
    meta_table = Table(metadata_data, colWidths=[100, 170, 100, 170])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))

    # 2. Executive Analytics
    if analytics_data:
        story.append(Paragraph("Executive Analytics & Summary Metrics", h2_style))
        kpis = analytics_data.get("kpis", [])
        if kpis:
            kpi_table_data = [
                [Paragraph(f"<b>{k.get('title', 'Metric')}</b>", body_style) for k in kpis[:4]],
                [Paragraph(str(k.get('value', 'N/A')), body_style) for k in kpis[:4]]
            ]
            kpi_table = Table(kpi_table_data, colWidths=[135] * len(kpis[:4]))
            kpi_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EEF2FF")),
                ('PADDING', (0, 0), (-1, -1), 6),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#C7D2FE")),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#C7D2FE")),
            ]))
            story.append(kpi_table)
            story.append(Spacer(1, 12))

        insights = analytics_data.get("insights", [])
        if insights:
            story.append(Paragraph("<b>Strategic Insights & Risk Highlights:</b>", body_style))
            for item in insights:
                itype = item.get("type", "positive").upper()
                text = item.get("text", "")
                bullet = f"• <b>[{itype}]</b> {text}"
                story.append(Paragraph(bullet, body_style))
            story.append(Spacer(1, 12))

    # 3. Agentic Outputs
    if agent_outputs:
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=10))
        story.append(Paragraph("Agentic Workflow Deliverables", h2_style))

        for title, output in agent_outputs.items():
            story.append(Paragraph(f"<b>{title}</b>", body_style))
            safe_output = html.escape(output)
            safe_output = safe_output.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")

            for para in safe_output.split("\n\n"):
                if para.strip():
                    clean_para = para.strip().replace("\n", "<br/>")
                    try:
                        story.append(Paragraph(clean_para, body_style))
                    except Exception:
                        plain_text = para.strip().replace("<br/>", " ")
                        story.append(Paragraph(html.escape(plain_text), body_style))
            story.append(Spacer(1, 10))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ------------------------------------------------------------------------------
# MULTI-PDF PROCESSING & VECTOR INDEXING PIPELINE
# ------------------------------------------------------------------------------
def process_multiple_pdfs(uploaded_files, chunk_size=1000, chunk_overlap=200):
    all_documents = []
    all_metadatas = []
    doc_catalog = {}
    total_tokens = 0
    total_pages = 0
    sample_text_list = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        pdf_bytes = uploaded_file.read()
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))

        file_pages = len(pdf_reader.pages)
        total_pages += file_pages
        file_text_length = 0

        for idx, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text() or ""
            words = page_text.split()
            file_text_length += len(words)

            if len(sample_text_list) < 15:
                sample_text_list.append(f"[{filename} - Pg {idx+1}]: {page_text[:500]}")

            chunks = text_splitter.split_text(page_text)
            for chunk in chunks:
                all_documents.append(chunk)
                all_metadatas.append({"source": filename, "page": idx + 1})

        total_tokens += file_text_length
        doc_catalog[filename] = {"pages": file_pages, "tokens": file_text_length}

    if not all_documents:
        return None, None, ""

    embeddings = load_embedding_model()
    vector_store = FAISS.from_texts(texts=all_documents, embedding=embeddings, metadatas=all_metadatas)

    stats = {
        "doc_count": len(uploaded_files),
        "catalog": doc_catalog,
        "total_pages": total_pages,
        "total_tokens": total_tokens,
        "total_chunks": len(all_documents)
    }
    corpus_sample = "\n\n".join(sample_text_list)[:15000]
    return vector_store, stats, corpus_sample


# ------------------------------------------------------------------------------
# SMALL UI HELPERS
# ------------------------------------------------------------------------------
def render_topbar(show_reset=False):
    col_brand, col_action = st.columns([4, 1])
    with col_brand:
        st.markdown(
            """
            <div class="topbar">
                <div class="topbar-brand">
                    <div class="topbar-logo">🧠</div>
                    <div>
                        <div class="topbar-title">IntellectPDF</div>
                        <div class="topbar-caption">Multi-Agent Intelligence Platform · IIT Patna Capstone</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if show_reset:
        with col_action:
            st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
            if st.button("🔄 Upload New Corpus / Reset", use_container_width=True):
                reset_corpus()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


def render_kpi(title, value):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(icon, title, subtitle):
    st.markdown(
        f"""
        <div class="section-header">
            <div class="section-icon">{icon}</div>
            <div>
                <p class="section-title">{title}</p>
                <p class="section-subtitle">{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight_badge(itype, text):
    css_class = f"insight-{itype}" if itype in ["positive", "attention", "critical"] else "insight-positive"
    icon = "✅" if itype == "positive" else ("⚠️" if itype == "attention" else "🚨")
    st.markdown(
        f"""<div class="insight-card {css_class}"><strong>{icon} {itype.upper()}:</strong> {text}</div>""",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------------
# API KEY GATE (NO FRONT-END INPUT FIELD — BACKEND SECRETS ONLY)
# ------------------------------------------------------------------------------
active_api_key = resolve_api_key()

if not active_api_key:
    render_topbar(show_reset=False)
    st.markdown(
        """
        <div class="config-warning-card">
            <div style="font-size:2.4rem; margin-bottom:10px;">🔐</div>
            <h3 style="margin-bottom:10px;">Gemini API Key Not Configured</h3>
            <p style="color:#64748B; font-size:0.95rem; line-height:1.6;">
                IntellectPDF requires a <b>GEMINI_API_KEY</b> to be configured on the server before it can run.
                This platform does not accept API keys from the browser for security reasons.
            </p>
            <p style="color:#64748B; font-size:0.88rem; line-height:1.6; margin-top:14px;">
                Add it to <code>.streamlit/secrets.toml</code> as:<br>
                <code>GEMINI_API_KEY = "your_key_here"</code><br>
                or export it as an environment variable <code>GEMINI_API_KEY</code> before launching the app.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

gemini_client = get_gemini_client(active_api_key)


# ==============================================================================
# LANDING PAGE (PRE-UPLOAD STATE)
# ==============================================================================
def render_landing_page():
    render_topbar(show_reset=False)

    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-badge"><span class="dot"></span> IIT Patna · Generative AI Capstone Project</div>
            <div class="hero-title">Turn dense PDFs into<br><span class="gradient">actionable intelligence</span></div>
            <p class="hero-subtitle">
                IntellectPDF is a multi-agent RAG platform that reads, indexes, compares, and interrogates your
                financial audits, medical reports, contracts, and research papers — with grounded, page-level
                citations on every answer.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("⚙️ Advanced RAG Configuration (optional)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.session_state.cfg_chunk_size = st.slider(
                "Chunk Size", 500, 2000, st.session_state.cfg_chunk_size, step=100
            )
        with c2:
            st.session_state.cfg_chunk_overlap = st.slider(
                "Chunk Overlap", 50, 400, st.session_state.cfg_chunk_overlap, step=25
            )
        with c3:
            st.session_state.cfg_top_k = st.slider(
                "Top-K Retrieval", 2, 10, st.session_state.cfg_top_k
            )

    st.markdown('<div class="upload-label">📤 Upload your document corpus</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="upload-sublabel">Drag and drop one or more PDF files, or click to browse</div>',
        unsafe_allow_html=True,
    )

    up_col1, up_col2, up_col3 = st.columns([1, 3, 1])
    with up_col2:
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"uploader_{st.session_state.uploader_version}",
        )

    if uploaded_files:
        current_names = [f.name for f in uploaded_files]
        if st.session_state.vector_store is None or st.session_state.file_list != current_names:
            with st.spinner("Indexing multi-document corpus into FAISS..."):
                v_store, stats, corpus_sample = process_multiple_pdfs(
                    uploaded_files,
                    st.session_state.cfg_chunk_size,
                    st.session_state.cfg_chunk_overlap,
                )
                if v_store:
                    st.session_state.vector_store = v_store
                    st.session_state.doc_stats = stats
                    st.session_state.file_list = current_names
                    st.session_state.corpus_sample = corpus_sample
                    st.session_state.analytics_data = None
                    st.session_state.quiz_data = None
                    st.session_state.chat_history = []
                    st.session_state.agent_outputs = {}
                    st.success(f"Indexed {len(uploaded_files)} PDF(s) successfully!")
                    st.rerun()
                else:
                    st.error("No readable text could be extracted from the uploaded file(s).")

    # ---- Feature Showcase ----
    # ---- Fixed Feature Showcase Layout ----
    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
    f_col1, f_col2, f_col3 = st.columns(3)
    
    with f_col1:
        st.markdown("""<div class="feature-card"><div class="feature-icon">📊</div><div class="feature-title">Executive Summary & Analytics</div><div class="feature-desc">Auto-extracted KPI cards, risk ratings, and interactive Plotly charts distilled straight from documents.</div></div>""", unsafe_allow_html=True)
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        st.markdown("""<div class="feature-card"><div class="feature-icon">🧩</div><div class="feature-title">Comprehension Engine</div><div class="feature-desc">Auto-generated 5-question quizzes with instant grading and explanation-backed feedback.</div></div>""", unsafe_allow_html=True)

    with f_col2:
        st.markdown("""<div class="feature-card"><div class="feature-icon">⚔️</div><div class="feature-title">Cross-Document Matrix</div><div class="feature-desc">Select multiple files to generate side-by-side comparison tables and strategic contrast reports.</div></div>""", unsafe_allow_html=True)
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        st.markdown("""<div class="feature-card"><div class="feature-icon">💬</div><div class="feature-title">Grounded RAG Chat</div><div class="feature-desc">Ask natural-language questions and get answers with exact page-level and file-level citations.</div></div>""", unsafe_allow_html=True)

    with f_col3:
        st.markdown("""<div class="feature-card"><div class="feature-icon">🚨</div><div class="feature-title">Anomaly & Risk Audit</div><div class="feature-desc">Automated detection of red flags, policy discrepancies, and compliance gaps across your corpus.</div></div>""", unsafe_allow_html=True)
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        st.markdown("""<div class="feature-card"><div class="feature-icon">📥</div><div class="feature-title">Report Exports</div><div class="feature-desc">One-click, professionally formatted PDF and Markdown exports built with ReportLab Platypus.</div></div>""", unsafe_allow_html=True)


# ==============================================================================
# DASHBOARD SECTIONS (POST-UPLOAD STATE)
# ==============================================================================
def render_section1_telemetry():
    with st.container(border=True):
        render_section_header("📡", "Ingestion & Corpus Telemetry", "Live statistics from the FAISS vector index")
        stats = st.session_state.doc_stats
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_kpi("Total Files", stats.get("doc_count", "N/A"))
        with c2:
            render_kpi("Total Pages", stats.get("total_pages", "N/A"))
        with c3:
            render_kpi("Est. Word Count", f"{stats.get('total_tokens', 0):,}")
        with c4:
            render_kpi("FAISS Vector Nodes", stats.get("total_chunks", "N/A"))

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📁 Per-File Breakdown"):
            for fname, meta in stats.get("catalog", {}).items():
                st.markdown(f"• **{fname}**: {meta['pages']} pages (~{meta['tokens']:,} words)")


def render_section2_analytics():
    with st.container(border=True):
        render_section_header("📊", "Executive Analytics & Visual Quantitative Insights",
                               "Gemini-extracted KPIs, risk badges, and chart-ready metrics")

        if st.session_state.analytics_data is None:
            if st.button("🚀 Run Executive Analytics Engine", type="primary"):
                with st.spinner("Extracting structured metrics and chart parameters..."):
                    system_prompt = """
                    Analyze the document text sample and generate a strict JSON response.
                    Respond ONLY with a valid JSON object matching this schema:
                    {
                      "kpis": [
                        {"title": "Overall Document Risk", "value": "Low Risk / Positive"},
                        {"title": "Primary Category", "value": "Financial / Medical / Research / Legal"},
                        {"title": "Key Entities Identified", "value": "12 Entities"},
                        {"title": "Core Topic", "value": "Primary Subject"}
                      ],
                      "insights": [
                        {"type": "positive", "text": "Positive observation text."},
                        {"type": "attention", "text": "Item requiring review text."},
                        {"type": "critical", "text": "High risk or critical warning text."}
                      ],
                      "chart": {
                        "chart_title": "Quantitative Distribution",
                        "x_axis_label": "Category",
                        "y_axis_label": "Value",
                        "series": [
                          {"label": "Metric A", "value": 45},
                          {"label": "Metric B", "value": 70},
                          {"label": "Metric C", "value": 25}
                        ]
                      }
                    }
                    """
                    try:
                        response = gemini_client.models.generate_content(
                            model=GEMINI_MODEL,
                            contents=[system_prompt, f"Document Corpus Sample:\n{st.session_state.corpus_sample}"],
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                temperature=0.2
                            )
                        )
                        cleaned_json = clean_json_response(response.text)
                        st.session_state.analytics_data = json.loads(cleaned_json)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Analytics Pipeline Error: {str(e)}")

        if st.session_state.analytics_data:
            analytics = st.session_state.analytics_data
            cols = st.columns(4)
            for idx, kpi in enumerate(analytics.get("kpis", [])):
                with cols[idx % 4]:
                    render_kpi(kpi.get("title", "Metric"), kpi.get("value", "N/A"))

            st.divider()
            col_left, col_right = st.columns([3, 2])

            with col_left:
                st.markdown("##### 📈 Visual Quantitative Insights")
                chart_info = analytics.get("chart", {})
                series_data = chart_info.get("series", [])
                if series_data:
                    df_chart = pd.DataFrame(series_data)
                    fig = px.bar(
                        df_chart, x="label", y="value", text="value",
                        title=chart_info.get("chart_title", "Corpus Data Breakdown"),
                        labels={"label": chart_info.get("x_axis_label", "Category"),
                                "value": chart_info.get("y_axis_label", "Value")},
                        color="value", color_continuous_scale="Blues"
                    )
                    fig.update_layout(
                        template="plotly_white",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#0F172A"),
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col_right:
                st.markdown("##### 🔍 Risk & Strategic Insights")
                for item in analytics.get("insights", []):
                    render_insight_badge(item.get("type", "positive"), item.get("text", ""))


def render_section3_comparative():
    with st.container(border=True):
        render_section_header("⚔️", "Cross-Document Matrix & Anomaly Detection",
                               "Side-by-side comparison, policy discrepancy audit, and risk flags")

        if len(st.session_state.file_list) < 2:
            st.info("💡 Upload 2 or more PDF documents to unlock cross-document comparative analysis.")
        else:
            selected_docs = st.multiselect(
                "Select Documents to Compare:",
                options=st.session_state.file_list,
                default=st.session_state.file_list[:2]
            )

            if st.button("⚖️ Generate Comparative Matrix", type="primary"):
                if len(selected_docs) < 2:
                    st.warning("Please select at least 2 documents to compare.")
                else:
                    with st.spinner("Executing multi-document comparison and anomaly audit..."):
                        retrieved_docs = st.session_state.vector_store.similarity_search(
                            "financial metrics revenue targets risks strategic goals compliance policy comparison",
                            k=8
                        )
                        context = "\n\n".join(
                            [f"[{d.metadata['source']} - Page {d.metadata['page']}]: {d.page_content}"
                             for d in retrieved_docs]
                        )

                        comp_prompt = f"""
                        Compare the following documents: {', '.join(selected_docs)}.
                        Provide a structured Markdown report containing:
                        1. Executive Summary of key contrasts.
                        2. Markdown Comparison Table with columns: (Parameter, {selected_docs[0]}, {selected_docs[1]}, Key Difference).
                        3. Anomaly Detection & Policy Discrepancy Flags — a bulleted list of inconsistencies,
                           missing clauses, contradictory figures, or compliance gaps found between the documents.
                        4. Critical Risks & Strategic Divergences.
                        """
                        try:
                            res = gemini_client.models.generate_content(
                                model=GEMINI_MODEL,
                                contents=[comp_prompt, f"Context Chunks:\n{context}"]
                            )
                            st.session_state["_comparative_result"] = res.text
                        except Exception as e:
                            st.error(f"Comparative Intelligence Error: {str(e)}")

            if st.session_state.get("_comparative_result"):
                st.divider()
                st.markdown(st.session_state["_comparative_result"])


def render_section4_agentic():
    with st.container(border=True):
        render_section_header("⚡", "Autonomous Multi-Agent Workflows & Executive Reports",
                               "Specialized agent routines plus one-click PDF / Markdown export")

        agent_cols = st.columns(4)
        with agent_cols[0]:
            run_swot = st.button("🎯 SWOT Agent", use_container_width=True)
        with agent_cols[1]:
            run_risk = st.button("🚨 Risk Audit", use_container_width=True)
        with agent_cols[2]:
            run_actions = st.button("📋 Action Items", use_container_width=True)
        with agent_cols[3]:
            run_translate = st.button("🌐 Translate (Hindi)", use_container_width=True)

        agent_target = None
        agent_prompt = ""
        if run_swot:
            agent_target = "SWOT Analysis"
            agent_prompt = "Perform a Strategic SWOT Analysis across the uploaded documents."
        elif run_risk:
            agent_target = "Risk & Anomaly Audit"
            agent_prompt = "Identify potential risks, compliance gaps, and red flags."
        elif run_actions:
            agent_target = "Action Item Tracker"
            agent_prompt = "Extract a clean table of key deliverables, action items, and deadlines."
        elif run_translate:
            agent_target = "Hindi Translation Summary"
            agent_prompt = "Provide an executive summary translated into fluent Hindi (Devanagari script)."

        if agent_target and agent_prompt:
            with st.spinner(f"Agent executing: {agent_target}..."):
                agent_instruction = f"You are a specialized enterprise AI Agent executing: {agent_target}.\nTask: {agent_prompt}"
                try:
                    agent_res = gemini_client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=[agent_instruction, f"Document Content:\n{st.session_state.corpus_sample}"],
                        config=types.GenerateContentConfig(temperature=0.2)
                    )
                    st.session_state.agent_outputs[agent_target] = agent_res.text
                except Exception as e:
                    st.error(f"Agent Execution Failure: {str(e)}")

        if st.session_state.agent_outputs:
            st.divider()
            for title, output in st.session_state.agent_outputs.items():
                st.markdown(f"##### {title}")
                st.markdown(output)
                st.divider()

        st.markdown("##### 📄 Export Full Executive Intelligence Report")
        dl_col1, dl_col2 = st.columns(2)

        doc_summary_title = ", ".join(st.session_state.file_list) if st.session_state.file_list else "Corpus"

        with dl_col1:
            pdf_bytes = generate_pdf_report(
                doc_summary_name=doc_summary_title,
                stats=st.session_state.doc_stats,
                analytics_data=st.session_state.analytics_data,
                agent_outputs=st.session_state.agent_outputs
            )
            st.download_button(
                label="📥 Download PDF Report (.pdf)",
                data=pdf_bytes,
                file_name="IntellectPDF_Executive_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        with dl_col2:
            report_content = "# IntellectPDF Executive Report\n"
            report_content += f"**Indexed Files:** {doc_summary_title}\n\n"

            if st.session_state.analytics_data:
                report_content += "## Executive Analytics Summary\n"
                for kpi in st.session_state.analytics_data.get("kpis", []):
                    report_content += f"- **{kpi.get('title')}:** {kpi.get('value')}\n"
                report_content += "\n"

            if st.session_state.agent_outputs:
                report_content += "## Agentic Workflow Deliverables\n"
                for title, output in st.session_state.agent_outputs.items():
                    report_content += f"### {title}\n{output}\n\n"

            st.download_button(
                label="📝 Download Markdown Report (.md)",
                data=report_content,
                file_name="IntellectPDF_Executive_Report.md",
                mime="text/markdown",
                use_container_width=True
            )


def render_section5_chat_and_quiz():
    with st.container(border=True):
        render_section_header("💬", "Deep RAG Chat & Comprehension Suite",
                               "Grounded conversational retrieval and self-assessment, side by side")

        tab_chat, tab_quiz = st.tabs(["💬 Deep RAG Chat", "📝 Assessment Engine"])

        # ---- Tab A: Deep RAG Chat ----
        with tab_chat:
            top_k = st.slider("Retrieval depth (Top-K chunks)", 2, 10, st.session_state.cfg_top_k, key="chat_top_k")
            st.session_state.cfg_top_k = top_k

            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "sources" in message and message["sources"]:
                        with st.expander("📚 Source Attribution & Citations"):
                            for src in message["sources"]:
                                st.markdown(f"📄 **File:** `{src['source']}` | **Page:** `{src['page']}`")
                                st.caption(f"_{src['text']}_")

            if user_query := st.chat_input("Ask a question across all uploaded documents..."):
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)

                with st.chat_message("assistant"):
                    with st.spinner("Searching vector index..."):
                        retrieved_docs = st.session_state.vector_store.similarity_search(user_query, k=top_k)

                        context_str = ""
                        sources_info = []
                        for rdoc in retrieved_docs:
                            src_file = rdoc.metadata.get("source", "Unknown")
                            page_num = rdoc.metadata.get("page", "Unknown")
                            context_str += f"\n--- [{src_file} - Page {page_num}] ---\n{rdoc.page_content}\n"
                            sources_info.append({
                                "source": src_file,
                                "page": page_num,
                                "text": rdoc.page_content[:250] + "..."
                            })

                        rag_prompt = f"""
                        Answer the user query strictly using the retrieved multi-document context below.
                        If information is absent, state clearly that it is not present in the documents.

                        Context:
                        {context_str}
                        """
                        try:
                            response = gemini_client.models.generate_content(
                                model=GEMINI_MODEL,
                                contents=[rag_prompt, f"User Query: {user_query}"],
                                config=types.GenerateContentConfig(temperature=0.1)
                            )
                            answer_text = response.text
                            st.markdown(answer_text)

                            with st.expander("📚 Source Attribution & Citations"):
                                for src in sources_info:
                                    st.markdown(f"📄 **File:** `{src['source']}` | **Page:** `{src['page']}`")
                                    st.caption(f"_{src['text']}_")

                            st.session_state.chat_history.append({
                                "role": "assistant", "content": answer_text, "sources": sources_info
                            })
                        except Exception as e:
                            st.error(f"RAG Inference Error: {str(e)}")

        # ---- Tab B: Assessment Engine ----
        with tab_quiz:
            if st.session_state.quiz_data is None:
                if st.button("📝 Generate Assessment Quiz", type="primary"):
                    with st.spinner("Generating 5 multiple choice questions..."):
                        quiz_prompt = """
                        Generate 5 distinct multiple choice questions based on the document corpus text.
                        Respond ONLY with a valid JSON array matching this schema:
                        [
                          {
                            "id": 1,
                            "question": "Question statement?",
                            "options": ["Option A", "Option B", "Option C", "Option D"],
                            "answer": "Option A",
                            "explanation": "Explanation justifying the answer."
                          }
                        ]
                        """
                        try:
                            response = gemini_client.models.generate_content(
                                model=GEMINI_MODEL,
                                contents=[quiz_prompt, f"Corpus Sample:\n{st.session_state.corpus_sample}"],
                                config=types.GenerateContentConfig(
                                    response_mime_type="application/json",
                                    temperature=0.3
                                )
                            )
                            cleaned_json = clean_json_response(response.text)
                            st.session_state.quiz_data = json.loads(cleaned_json)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Quiz Generation Error: {str(e)}")

            if st.session_state.quiz_data:
                with st.form("quiz_form"):
                    user_answers = {}
                    for idx, q in enumerate(st.session_state.quiz_data):
                        st.markdown(f"**Q{idx+1}: {q['question']}**")
                        user_answers[idx] = st.radio(
                            f"Q{idx+1} Options:",
                            options=q["options"],
                            key=f"quiz_q_{idx}",
                            index=None,
                            label_visibility="collapsed"
                        )
                        st.divider()
                    submitted = st.form_submit_button("Submit Assessment")

                if submitted:
                    score = 0
                    total = len(st.session_state.quiz_data)
                    st.markdown("##### 🏆 Quiz Assessment Results")
                    for idx, q in enumerate(st.session_state.quiz_data):
                        selected = user_answers.get(idx)
                        correct = q["answer"]
                        if selected == correct:
                            score += 1
                            st.success(f"**Question {idx+1}: Correct!**")
                        else:
                            st.error(f"**Question {idx+1}: Incorrect.** (Your Answer: `{selected}` | Correct: `{correct}`)")
                        st.markdown(f"**Reasoning:** {q['explanation']}")
                    st.metric(label="Final Score", value=f"{score} / {total}", delta=f"{(score/total)*100:.0f}% Accuracy")

def render_footer():
    footer_html = """
    <div class="app-footer">
        <div class="footer-left">
            Developed with ❤️ by <strong>Nishant Raj</strong> · IntellectPDF Platform
        </div>
        <div class="footer-socials">
            <!-- LinkedIn -->
            <a href="https://www.linkedin.com/in/nishant-raj-82972b208/" target="_blank" class="social-btn" title="LinkedIn">
                <svg viewBox="0 0 24 24"><path d="M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14m-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.28 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93h2.75M6.88 8.56a1.68 1.68 0 0 0 1.68-1.68c0-.93-.75-1.69-1.68-1.69a1.69 1.69 0 0 0-1.69 1.69c0 .93.76 1.68 1.69 1.68m1.39 9.94v-8.37H5.5v8.37h2.77z"/></svg>
            </a>
            <!-- GitHub -->
            <a href="https://github.com/thenishantraj" target="_blank" class="social-btn" title="GitHub">
                <svg viewBox="0 0 24 24"><path d="M12 2A10 10 0 0 0 2 12c0 4.42 2.87 8.17 6.84 9.5.5.08.66-.23.66-.5v-1.69c-2.77.6-3.36-1.34-3.36-1.34-.46-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.6.07-.6 1 .07 1.53 1.03 1.53 1.03.87 1.52 2.34 1.07 2.91.83.1-.65.35-1.09.63-1.34-2.22-.25-4.55-1.11-4.55-4.92 0-1.11.38-2 1.03-2.71-.1-.25-.45-1.29.1-2.64 0 0 .84-.27 2.75 1.02.79-.22 1.65-.33 2.5-.33.85 0 1.71.11 2.5.33 1.91-1.29 2.75-1.02 2.75-1.02.55 1.35.2 2.39.1 2.64.65.71 1.03 1.6 1.03 2.71 0 3.82-2.34 4.66-4.57 4.91.36.31.69.92.69 1.85V21c0 .27.16.59.67.5C19.14 20.16 22 16.42 22 12A10 10 0 0 0 12 2z"/></svg>
            </a>
            <!-- Email / Contact -->
            <a href="mailto:nishantraj6581@gmail.com" class="social-btn" title="Email Contact">
                <svg viewBox="0 0 24 24"><path d="M20 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 4l-8 5-8-5V6l8 5 8-5v2z"/></svg>
            </a>
        </div>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True) 

def render_dashboard():
    render_topbar(show_reset=True)
    st.markdown("<br>", unsafe_allow_html=True)
    render_section1_telemetry()
    render_section2_analytics()
    render_section3_comparative()
    render_section4_agentic()
    render_section5_chat_and_quiz()


# ==============================================================================
# MAIN ROUTING
# ==============================================================================
if st.session_state.vector_store is None:
    render_landing_page()
else:
    render_dashboard()
render_footer()
