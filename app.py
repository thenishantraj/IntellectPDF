import json
import re
import os
import io
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
from reportlab.lib.enums import TA_LEFT

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION & ENTERPRISE DARK / GLASSMORPHIC CSS
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="IntellectPDF | IIT Patna Capstone",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    .stApp {
        background-color: #0E1117;
        color: #E0E6ED;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .kpi-card {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #4F46E5;
        border-radius: 8px;
        padding: 16px;
        text-align: left;
    }
    .kpi-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        color: #9CA3AF;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F9FAFB;
    }

    .insight-card {
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 12px;
        font-size: 0.95rem;
        border-left: 5px solid;
    }
    .insight-positive {
        background: rgba(16, 185, 129, 0.1);
        border-color: #10B981;
        color: #D1FAE5;
    }
    .insight-attention {
        background: rgba(245, 158, 11, 0.1);
        border-color: #F59E0B;
        color: #FEF3C7;
    }
    .insight-critical {
        background: rgba(239, 68, 68, 0.1);
        border-color: #EF4444;
        color: #FEE2E2;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ------------------------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}
if "analytics_data" not in st.session_state:
    st.session_state.analytics_data = None
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent_outputs" not in st.session_state:
    st.session_state.agent_outputs = {}
if "file_list" not in st.session_state:
    st.session_state.file_list = []
if "corpus_sample" not in st.session_state:
    st.session_state.corpus_sample = ""

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS & GEMINI CLIENT SETUP
# ------------------------------------------------------------------------------
def get_gemini_client(api_key: str):
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def clean_json_response(raw_response: str) -> str:
    raw_response = raw_response.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_response

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ------------------------------------------------------------------------------
# REPORTLAB PDF GENERATOR FUNCTION
# ------------------------------------------------------------------------------
def generate_pdf_report(doc_summary_name, stats, analytics_data, agent_outputs):
    """Generates an executive report PDF dynamically using ReportLab."""
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
    ACCENT_COLOR = colors.HexColor("#4F46E5")
    TEXT_COLOR = colors.HexColor("#334155")
    BG_LIGHT = colors.HexColor("#F8FAFC")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=PRIMARY_COLOR,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=15
    )

    h2_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=ACCENT_COLOR,
        spaceBefore=12,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=TEXT_COLOR,
        spaceAfter=6
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
            kpi_table = Table(kpi_table_data, colWidths=[135]*len(kpis[:4]))
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
            clean_output = output.replace("#", "").replace("**", "<b>").replace("*", "")
            for para in clean_output.split("\n\n"):
                if para.strip():
                    story.append(Paragraph(para.strip().replace("\n", "<br/>"), body_style))
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
                all_metadatas.append({
                    "source": filename,
                    "page": idx + 1
                })

        total_tokens += file_text_length
        doc_catalog[filename] = {
            "pages": file_pages,
            "tokens": file_text_length
        }

    if not all_documents:
        return None, None, "No readable text extracted from uploaded files.", ""

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
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric-headers/100/brain.png", width=60)
    st.title("IntellectPDF")
    st.caption("IIT Patna Capstone | Multi-Doc Agentic RAG")
    st.divider()

    secret_key = st.secrets.get("GEMINI_API_KEY", "")
    api_key_input = st.text_input("Gemini API Key", value=secret_key, type="password")
    
    active_api_key = api_key_input if api_key_input else secret_key
    gemini_client = get_gemini_client(active_api_key)

    st.divider()
    st.subheader("⚙️ RAG Engine Settings")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", 50, 400, 200, step=25)
    top_k = st.slider("Top-K Retrieval", 2, 10, 4)

    st.divider()
    uploaded_files = st.file_uploader(
        "Upload Target Document(s)", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        current_names = [f.name for f in uploaded_files]
        if st.session_state.vector_store is None or st.session_state.file_list != current_names:
            with st.spinner("Indexing multi-document corpus into FAISS..."):
                v_store, stats, corpus_sample = process_multiple_pdfs(uploaded_files, chunk_size, chunk_overlap)
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

    if st.session_state.doc_stats:
        st.divider()
        st.markdown("### 📊 Document Metadata")
        st.markdown(f"**Total Files:** `{st.session_state.doc_stats['doc_count']}`")
        st.markdown(f"**Total Pages:** `{st.session_state.doc_stats['total_pages']}`")
        st.markdown(f"**Est. Word Count:** `{st.session_state.doc_stats['total_tokens']:,}`")
        st.markdown(f"**FAISS Vector Nodes:** `{st.session_state.doc_stats['total_chunks']}`")
        
        with st.expander("📁 File Breakdown"):
            for fname, meta in st.session_state.doc_stats["catalog"].items():
                st.markdown(f"• **{fname}**: {meta['pages']} pages (~{meta['tokens']:,} words)")

# ------------------------------------------------------------------------------
# MAIN APPLICATION WORKSPACE
# ------------------------------------------------------------------------------
st.title("🧠 IntellectPDF: Multi-Agent Intelligence Platform")

if not active_api_key:
    st.warning("⚠️ Please enter a Gemini API Key in the sidebar or setup `.streamlit/secrets.toml` to continue.")
    st.stop()

if st.session_state.vector_store is None:
    st.info("👈 Upload one or more PDF documents in the sidebar to initialize the AI analysis pipeline.")
    st.markdown("""
    <div class="glass-card">
        <h3>Welcome to IntellectPDF</h3>
        <p>An enterprise multi-agent analytics engine designed to extract, synthesize, visualize, and interact with complex unstructured documents.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Summary & Analytics",
    "⚔️ Comparative Intelligence",
    "📝 Assessment Engine",
    "💬 Deep RAG Chat",
    "⚡ Agentic Actions & Reports"
])

# ------------------------------------------------------------------------------
# TAB 1: EXECUTIVE SUMMARY & ANALYTICS
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Executive Document Dashboard")

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
                        model='gemini-3.6-flash',
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
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">{kpi.get('title', 'Metric')}</div>
                    <div class="kpi-value">{kpi.get('value', 'N/A')}</div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("### 📈 Visual Quantitative Insights")
            chart_info = analytics.get("chart", {})
            series_data = chart_info.get("series", [])

            if series_data:
                df_chart = pd.DataFrame(series_data)
                fig = px.bar(
                    df_chart,
                    x="label",
                    y="value",
                    text="value",
                    title=chart_info.get("chart_title", "Corpus Data Breakdown"),
                    labels={"label": chart_info.get("x_axis_label", "Category"), "value": chart_info.get("y_axis_label", "Value")},
                    color="value",
                    color_continuous_scale="Viridis"
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("### 🔍 Risk & Strategic Insights")
            for item in analytics.get("insights", []):
                itype = item.get("type", "positive")
                css_class = f"insight-{itype}" if itype in ["positive", "attention", "critical"] else "insight-positive"
                icon = "✅" if itype == "positive" else ("⚠️" if itype == "attention" else "🚨")
                st.markdown(f"""
                <div class="insight-card {css_class}">
                    <strong>{icon} {itype.upper()}:</strong> {item.get('text', '')}
                </div>
                """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 2: COMPARATIVE INTELLIGENCE MATRIX
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Cross-Document Comparative Intelligence Matrix")

    if len(st.session_state.file_list) < 2:
        st.info("💡 Upload 2 or more PDF documents in the sidebar to run cross-document comparative analysis.")
    else:
        selected_docs = st.multiselect(
            "Select Documents to Compare:", 
            options=st.session_state.file_list, 
            default=st.session_state.file_list[:2]
        )

        if st.button("⚖️ Generate Comparative Matrix", type="primary"):
            with st.spinner("Executing multi-document comparison..."):
                retrieved_docs = st.session_state.vector_store.similarity_search("financial metrics revenue targets risks strategic goals comparison", k=8)
                context = "\n\n".join([f"[{d.metadata['source']} - Page {d.metadata['page']}]: {d.page_content}" for d in retrieved_docs])

                comp_prompt = f"""
                Compare the following documents: {', '.join(selected_docs)}.
                Provide a structured Markdown report containing:
                1. Executive Summary of key contrasts.
                2. Markdown Comparison Table with columns: (Parameter, {selected_docs[0]}, {selected_docs[1]}, Key Difference).
                3. Critical Risks & Strategic Divergences.
                """

                res = gemini_client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=[comp_prompt, f"Context Chunks:\n{context}"]
                )
                st.markdown(res.text)

# ------------------------------------------------------------------------------
# TAB 3: ASSESSMENT ENGINE
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Document Comprehension & Assessment Engine")

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
                        model='gemini-3.6-flash',
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
            st.markdown("### 🏆 Quiz Assessment Results")
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

# ------------------------------------------------------------------------------
# TAB 4: DEEP RAG CHAT
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Deep RAG Chat with Citation Tracking")

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
                for doc in retrieved_docs:
                    src_file = doc.metadata.get("source", "Unknown")
                    page_num = doc.metadata.get("page", "Unknown")
                    context_str += f"\n--- [{src_file} - Page {page_num}] ---\n{doc.page_content}\n"
                    sources_info.append({
                        "source": src_file,
                        "page": page_num,
                        "text": doc.page_content[:250] + "..."
                    })

                rag_prompt = f"""
                Answer the user query strictly using the retrieved multi-document context below.
                If information is absent, state clearly that it is not present in the documents.

                Context:
                {context_str}
                """

                try:
                    response = gemini_client.models.generate_content(
                        model='gemini-3.6-flash',
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
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources_info
                    })
                except Exception as e:
                    st.error(f"RAG Inference Error: {str(e)}")

# ------------------------------------------------------------------------------
# TAB 5: AGENTIC ACTIONS & REPORT GENERATOR
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("Autonomous Multi-Agent Task Routines")

    agent_cols = st.columns(4)
    with agent_cols[0]:
        run_swot = st.button("🎯 SWOT Analysis Agent", use_container_width=True)
    with agent_cols[1]:
        run_risk = st.button("🚨 Anomaly & Risk Audit", use_container_width=True)
    with agent_cols[2]:
        run_actions = st.button("📋 Action Item Tracker", use_container_width=True)
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
                    model='gemini-3.6-flash',
                    contents=[agent_instruction, f"Document Content:\n{st.session_state.corpus_sample}"],
                    config=types.GenerateContentConfig(temperature=0.2)
                )
                st.session_state.agent_outputs[agent_target] = agent_res.text
            except Exception as e:
                st.error(f"Agent Execution Failure: {str(e)}")

    if st.session_state.agent_outputs:
        st.divider()
        for title, output in st.session_state.agent_outputs.items():
            st.markdown(f"### {title}")
            st.markdown(output)
            st.divider()

    st.markdown("### 📄 Export Full Executive Intelligence Report")
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
        report_content = f"# IntellectPDF Executive Report\n"
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
