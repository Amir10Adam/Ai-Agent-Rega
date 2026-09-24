"""
Stakeholder Data Chat App (Fixed Duplicate Headers + Token Tracking Edition)
-----------------------------------------------------------
"""

import os
import re
import time
import warnings
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from groq import Groq
import gspread
from google.oauth2.service_account import Credentials

# ==========================================
# 1. إعدادات الاعتمادات (تدعم المحلي والسحابي أماناً)
# ==========================================
scopes = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

creds = None
try:
    if "gcp_service_account" in st.secrets:
        service_account_info = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
except Exception:
    pass

if not creds:
    GOOGLE_SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "credentials.json")
    creds = Credentials.from_service_account_file(GOOGLE_SERVICE_ACCOUNT_JSON, scopes=scopes)

# ==========================================
GOOGLE_SHEET_ID = st.secrets.get("GOOGLE_SHEET_ID", os.environ.get("GOOGLE_SHEET_ID"))
GOOGLE_WORKSHEET_NAME = st.secrets.get("GOOGLE_WORKSHEET_NAME", os.environ.get("GOOGLE_WORKSHEET_NAME", "all_data"))
DATASET_1_LABEL = st.secrets.get("DATASET_1_LABEL", os.environ.get("DATASET_1_LABEL", "real_estate_all_data"))

GOOGLE_SHEET_ID_2 = st.secrets.get("GOOGLE_SHEET_ID_2", os.environ.get("GOOGLE_SHEET_ID_2"))
GOOGLE_WORKSHEET_NAME_2 = st.secrets.get("GOOGLE_WORKSHEET_NAME_2", os.environ.get("GOOGLE_WORKSHEET_NAME_2", "city"))
DATASET_2_LABEL = st.secrets.get("DATASET_2_LABEL", os.environ.get("DATASET_2_LABEL", "real_estate_city"))   # ← شيلت المسافة الزيادة بعد city

# الشيت الثاني / مسار مكة (وتاباته)
GOOGLE_SHEET_ID_3 = st.secrets.get("GOOGLE_SHEET_ID_3", os.environ.get("GOOGLE_SHEET_ID_3"))
GOOGLE_WORKSHEET_NAME_3 = st.secrets.get("GOOGLE_WORKSHEET_NAME_3", os.environ.get("GOOGLE_WORKSHEET_NAME_3", "Units Details"))
DATASET_3_LABEL = st.secrets.get("DATASET_3_LABEL", os.environ.get("DATASET_3_LABEL", "masar_makkah_units"))

GOOGLE_WORKSHEET_NAME_4 = st.secrets.get("GOOGLE_WORKSHEET_NAME_4", os.environ.get("GOOGLE_WORKSHEET_NAME_4", "Change log Oracle Format"))
DATASET_4_LABEL = st.secrets.get("DATASET_4_LABEL", os.environ.get("DATASET_4_LABEL", "masar_makkah_changelog"))

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
GROQ_MODEL = st.secrets.get("GROQ_MODEL", os.environ.get("GROQ_MODEL", "llama3-8b-8192"))

MAX_AGENT_STEPS = 12

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about multiple datasets/tabs "
    "(real_estate_all_data, real_estate_city, masar_makkah_units, masar_makkah_changelog) "
    "for a business stakeholder. Always pick the correct `dataset` name. Answer in clear, "
    "natural language (Arabic if the user asked in Arabic, English otherwise), and never "
    "show raw code.\n\n"
    "DATASET SELECTION RULES for real estate questions:\n"
    "1. If the question asks for a CITY-LEVEL summary (total transactions, total value, "
    "average price, comparing cities) with NO breakdown by property type, neighborhood, "
    "or month requested -> use 'real_estate_city' FIRST, since it already holds "
    "pre-aggregated per-city numbers and is faster/more reliable for that.\n"
    "2. If the requested city, month, or metric is NOT found in 'real_estate_city' -> "
    "fall back to 'real_estate_all_data' and compute it yourself by filtering/grouping the "
    "raw transactions. Explicitly tell the user the number came from the raw data because "
    "it wasn't available in the city summary.\n"
    "3. If the question asks for a breakdown BY PROPERTY TYPE (نوع العقار), BY "
    "NEIGHBORHOOD (حي), or BY MONTH -> always use 'real_estate_all_data' directly, since "
    "'real_estate_city' has no such columns — never try 'real_estate_city' for these first.\n"
    "4. Never silently substitute a different aggregation than what was asked (e.g. don't "
    "return a city-wide total when the user asked for a per-neighborhood or per-type "
    "breakdown). If a requested column or breakdown genuinely doesn't exist in the chosen "
    "dataset, say so clearly instead of returning a misleading number.\n"
    "5. Always tell the user which dataset ('real_estate_city' or 'real_estate_all_data') "
    "the answer came from, so they know the source."
)


# ==========================================
# 3. تتبع استهلاك الـ tokens
# ==========================================

def extract_token_usage(response) -> dict:
    usage = getattr(response, "usage_metadata", None)
    if usage:
        return {
            "input": usage.get("input_tokens", 0),
            "output": usage.get("output_tokens", 0),
            "total": usage.get("total_tokens", 0),
        }
    meta = getattr(response, "response_metadata", {}) or {}
    token_usage = meta.get("token_usage", {})
    if token_usage:
        return {
            "input": token_usage.get("prompt_tokens", 0),
            "output": token_usage.get("completion_tokens", 0),
            "total": token_usage.get("total_tokens", 0),
        }
    return {"input": 0, "output": 0, "total": 0}


def accumulate_token_usage(response):
    usage = extract_token_usage(response)
    st.session_state.setdefault("session_tokens", {"input": 0, "output": 0, "total": 0})
    st.session_state.session_tokens["input"] += usage["input"]
    st.session_state.session_tokens["output"] += usage["output"]
    st.session_state.session_tokens["total"] += usage["total"]
    
def record_latency(elapsed: float):
    st.session_state.setdefault("latency_log", [])
    st.session_state.latency_log.append(elapsed)

def check_groq_rate_limits(model: str = GROQ_MODEL) -> dict:
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        headers = response.headers
        return {
            "limit_tokens": headers.get("x-ratelimit-limit-tokens"),
            "remaining_tokens": headers.get("x-ratelimit-remaining-tokens"),
            "reset_tokens": headers.get("x-ratelimit-reset-tokens"),
            "limit_requests": headers.get("x-ratelimit-limit-requests"),
            "remaining_requests": headers.get("x-ratelimit-remaining-requests"),
            "reset_requests": headers.get("x-ratelimit-reset-requests"),
        }
    except Exception as e:
        return {"error": str(e)}


def render_token_sidebar():
    with st.sidebar:
        st.subheader("📊 استهلاك التوكنز")

        session_tokens = st.session_state.get("session_tokens", {"input": 0, "output": 0, "total": 0})
        st.metric("إجمالي التوكنز المستهلكة في الجلسة", session_tokens["total"])
        st.caption(f"مدخلة: {session_tokens['input']} | مخرجة: {session_tokens['output']}")
        # ---- ضيف الجزء ده هنا ----
        latency_log = st.session_state.get("latency_log", [])
        if latency_log:
            avg_latency = sum(latency_log) / len(latency_log)
            last_latency = latency_log[-1]
            st.metric("متوسط زمن الاستجابة", f"{avg_latency:.2f} ثانية")
            st.caption(f"آخر سؤال: {last_latency:.2f} ثانية | عدد الأسئلة: {len(latency_log)}")
        # ---- لحد هنا ----
        st.divider()
        if st.button("تحقق من الحد المتبقي عند Groq"):
            with st.spinner("بجيب البيانات من Groq..."):
                info = check_groq_rate_limits()
            if "error" in info:
                st.error(f"تعذّر جلب البيانات: {info['error']}")
            else:
                st.metric(
                    "Tokens المتبقية (في الدقيقة)",
                    info.get("remaining_tokens", "—"),
                    help=f"الحد الكلي: {info.get('limit_tokens', '—')} | يتصفّر بعد: {info.get('reset_tokens', '—')}",
                )
                st.metric(
                    "Requests المتبقية (يوميًا)",
                    info.get("remaining_requests", "—"),
                    help=f"الحد الكلي: {info.get('limit_requests', '—')} | يتصفّر بعد: {info.get('reset_requests', '—')}",
                )
                st.caption("ملاحظة: الفحص نفسه بيستهلك توكن واحد تقريبًا.")


# ==========================================
# 4. دالة تحميل الشيتات
# ==========================================

@st.cache_data(ttl=600, show_spinner="Loading data from Google Sheets tabs...")
def load_sheet_as_dataframe(sheet_id: str, worksheet_name: str) -> pd.DataFrame:
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id).worksheet(worksheet_name)

    data = sheet.get_all_values()
    if not data or len(data) <= 1:
        raise ValueError(f"The worksheet '{worksheet_name}' is empty or has no data.")

    headers = data[0]
    seen = {}
    unique_headers = []
    for h in headers:
        h_str = str(h).strip()
        if not h_str:
            h_str = "Unnamed"
        if h_str in seen:
            seen[h_str] += 1
            unique_headers.append(f"{h_str}_{seen[h_str]}")
        else:
            seen[h_str] = 0
            unique_headers.append(h_str)

    df = pd.DataFrame(data[1:], columns=unique_headers)
    return df


FORBIDDEN_TOKENS = ("import ", "__", "open(", "exec(", "eval(", "os.", "sys.", "subprocess")


def _extract_code(raw_text: str) -> str:
    text = raw_text.strip()
    fence_match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    return text.strip()


def _safe_exec_pandas(code: str, df: pd.DataFrame):
    if any(tok in code for tok in FORBIDDEN_TOKENS):
        raise ValueError("Generated code contains a disallowed operation.")
    safe_builtins = {"len": len, "range": range, "min": min, "max": max, "sum": sum, "sorted": sorted, "round": round, "abs": abs, "list": list, "dict": dict}
    local_vars = {"df": df, "pd": pd}
    exec(code, {"__builtins__": safe_builtins}, local_vars)
    return local_vars.get("result", "Code executed but did not set a `result` variable.")


def make_pandas_tool(datasets: dict[str, pd.DataFrame], code_llm):
    dataset_summaries = "\n".join(f'- "{name}": columns = {list(d.columns)}, shape = {d.shape}' for name, d in datasets.items())
    dataset_names = list(datasets.keys())

    @tool
    def execute_pandas_query(question_description: str, dataset: str) -> str:
        """Answer statistical, numerical, aggregation, or filtering questions using pandas."""
        if dataset not in datasets:
            return f"Unknown dataset '{dataset}'. Valid options are: {dataset_names}"
        df = datasets[dataset]
        
        prompt = f"""You are a pandas expert. DataFrame `df` is loaded. Columns: {list(df.columns)}.
Question: "{question_description}"

Write ONLY executable Python (pandas) code assigning the answer to a variable named `result`.
- CRITICAL INSTRUCTION: Never drop, filter out, or ignore outliers, extreme values, or high values unless the user explicitly and directly asks you to remove outliers. Always include all data points.
- IMPORTANT: every column in `df` was loaded as plain text/strings, even numeric-looking ones.
  Always convert numeric columns first with pd.to_numeric(df[col], errors="coerce") before any
  math, comparison, sum, mean, or sorting operation on them.
- When parsing dates with pd.to_datetime, always pass format='mixed' to avoid parsing warnings.
- Do not import anything. No markdown fences."""
        
        try:
            response = code_llm.invoke(prompt)
            accumulate_token_usage(response)
            code = _extract_code(response.content if hasattr(response, "content") else str(response))
            return str(_safe_exec_pandas(code, df))
        except Exception as e:
            return f"Error: {e}"

    execute_pandas_query.description = f"Run pandas query. Dataset must be one of: {dataset_names}.\n{dataset_summaries}"
    return execute_pandas_query


def run_agent_turn(agent_llm, tools_by_name: dict, chat_history: list, question: str) -> tuple[str, float]:
    start_time = time.time()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + chat_history + [HumanMessage(content=question)]
    for _ in range(MAX_AGENT_STEPS):
        response = agent_llm.invoke(messages)
        accumulate_token_usage(response)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            elapsed = time.time() - start_time
            return (response.content or "لم يتمكن النموذج من توليد إجابة."), elapsed 
        for call in tool_calls:
            selected_tool = tools_by_name.get(call["name"])
            tool_output = selected_tool.invoke(call["args"]) if selected_tool else f"Unknown tool: {call['name']}"
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=call["id"]))
            elapsed = time.time() - start_time
    return "توقفت بعد عدة محاولات.",elapsed


st.set_page_config(page_title="اسأل عن البيانات", page_icon="📊")
st.title("📊 اسأل عن بيانات الشيتات والتابات المختلفة")

if not GOOGLE_SHEET_ID or not GROQ_API_KEY or not GROQ_MODEL:
    st.error("تأكد من إعداد المتغيرات الأساسية (GOOGLE_SHEET_ID, GROQ_API_KEY, GROQ_MODEL) في الـ Secrets أو ملف .env")
    st.stop()

render_token_sidebar()

try:
    datasets = {}

    # تحميل التاب الأول من الشيت الأول
    if GOOGLE_SHEET_ID and GOOGLE_WORKSHEET_NAME:
        datasets[DATASET_1_LABEL] = load_sheet_as_dataframe(GOOGLE_SHEET_ID, GOOGLE_WORKSHEET_NAME)

    # تحميل التاب التاني من شيت تاني منفصل (city)
    if GOOGLE_SHEET_ID_2 and GOOGLE_WORKSHEET_NAME_2:
        datasets[DATASET_2_LABEL] = load_sheet_as_dataframe(GOOGLE_SHEET_ID_2, GOOGLE_WORKSHEET_NAME_2)

    # تحميل التابتين من الشيت الثالث (مسار مكة)
    if GOOGLE_SHEET_ID_3 and GOOGLE_WORKSHEET_NAME_3:
        datasets[DATASET_3_LABEL] = load_sheet_as_dataframe(GOOGLE_SHEET_ID_3, GOOGLE_WORKSHEET_NAME_3)
    if GOOGLE_SHEET_ID_3 and GOOGLE_WORKSHEET_NAME_4:
        datasets[DATASET_4_LABEL] = load_sheet_as_dataframe(GOOGLE_SHEET_ID_3, GOOGLE_WORKSHEET_NAME_4)

    if not GOOGLE_SHEET_ID_2:
        st.warning("⚠️ GOOGLE_SHEET_ID_2 غير محدد — لم يتم تحميل بيانات city.")
    if not GOOGLE_SHEET_ID_3:
        st.warning("⚠️ GOOGLE_SHEET_ID_3 غير محدد — لم يتم تحميل بيانات مسار مكة.")

except Exception as e:
    st.error(f"خطأ في تحميل التابات: {e}")
    st.stop()
# عرض تقرير سريع بالتابات المحملة على الواجهة
for name, d in datasets.items():
    st.caption(f"📁 **Dataset (`{name}`)**: تم تحميل {len(d)} صف و {len(d.columns)} عمود.")

llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY)
pandas_tool = make_pandas_tool(datasets, llm)
tools_by_name = {pandas_tool.name: pandas_tool}
agent_llm = llm.bind_tools([pandas_tool])

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.lc_history = []

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

question = st.chat_input("اسأل سؤال عن البيانات في أي تابة...")
if question:
    st.session_state.messages.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("جاري المعالجة..."):
            answer, elapsed = run_agent_turn(agent_llm, tools_by_name, st.session_state.lc_history, question)
            record_latency(elapsed)
        st.markdown(answer)
        st.caption(f"⏱️ استغرقت {elapsed:.2f} ثانية")   # اختياري: تعرض الوقت تحت كل رد

    st.session_state.messages.append(("assistant", answer))
    st.session_state.lc_history.extend([HumanMessage(content=question), AIMessage(content=answer)])