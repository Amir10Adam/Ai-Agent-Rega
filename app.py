"""
Stakeholder Data Chat App (Fixed Duplicate Headers + Token Tracking Edition)
-----------------------------------------------------------
"""

import os
import re
import warnings
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning)

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from groq import Groq  # نستخدمها فقط لقراءة حدود الاستهلاك (rate limits) من الـ headers
import gspread
from google.oauth2.service_account import Credentials
# ==========================================
# إعدادات الاعتمادات (تدعم المحلي والسحابي أماناً)
# ==========================================
scopes = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

# التحقق مما إذا كنا نعمل على منصة Streamlit Cloud (عبر الـ Secrets)
if "gcp_service_account" in st.secrets:
    service_account_info = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
else:
    # محلياً على جهازك باستخدام ملف credentials.json
    GOOGLE_SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "credentials.json")
    creds = Credentials.from_service_account_file(GOOGLE_SERVICE_ACCOUNT_JSON, scopes=scopes)

# دالة تحميل الشيتات والتابات
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

load_dotenv()

# ==========================================
# 1. إعدادات المتغيرات والشيتات
# ==========================================
GOOGLE_SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "credentials.json")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# لازم قيمة افتراضية: لو الـ .env نسي يحددها، الكود كان بيفشل بـ error غامض بدل رسالة واضحة.
# llama-3.3-70b-versatile اتقفل من Groq في 16 أغسطس 2026؛ البديل الرسمي الموصى به هو ده:
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# الشيت الأول (وتاباته)
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")
GOOGLE_WORKSHEET_NAME = os.environ.get("GOOGLE_WORKSHEET_NAME", "all_data")
DATASET_1_LABEL = os.environ.get("DATASET_1_LABEL", "real_estate_all_data")

GOOGLE_WORKSHEET_NAME_2 = os.environ.get("GOOGLE_WORKSHEET_NAME_2", "city")
DATASET_2_LABEL = os.environ.get("DATASET_2_LABEL", "real_estate_city")

# الشيت الثاني / مسار مكة (وتاباته)
GOOGLE_SHEET_ID_3 = os.environ.get("GOOGLE_SHEET_ID_3")
GOOGLE_WORKSHEET_NAME_3 = os.environ.get("GOOGLE_WORKSHEET_NAME_3", "Units Details")
DATASET_3_LABEL = os.environ.get("DATASET_3_LABEL", "masar_makkah_units")

GOOGLE_WORKSHEET_NAME_4 = os.environ.get("GOOGLE_WORKSHEET_NAME_4", "Change log Oracle Format")
DATASET_4_LABEL = os.environ.get("DATASET_4_LABEL", "masar_makkah_changelog")

MAX_AGENT_STEPS = 6

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about multiple datasets/tabs "
    "(including real_estate_all_data, real_estate_city, masar_makkah_units, masar_makkah_changelog) "
    "for a business stakeholder. Always pick the correct `dataset` name. Answer in clear, natural language "
    "(Arabic if the user asked in Arabic, English otherwise), and never show raw code."
)


# ==========================================
# 2. تتبع استهلاك الـ tokens
# ==========================================

def extract_token_usage(response) -> dict:
    """يسحب عدد الـ tokens من رد الموديل (بدون أي طلب إضافي)."""
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


def check_groq_rate_limits(model: str = GROQ_MODEL) -> dict:
    """
    يبعت رسالة صغيرة جدًا للـ API الرسمي عشان يقرأ الـ headers اللي فيها حدود
    الاستهلاك الحالية عند Groq (TPM / RPD). ده بيستهلك تقريبًا توكن واحد، فمخصص
    لزرار "تحقق" وليس لكل رسالة شات.
    """
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
# 3. تحميل الشيتات
# ==========================================

@st.cache_data(ttl=600, show_spinner="Loading data from Google Sheets tabs...")
def load_sheet_as_dataframe(sheet_id: str, worksheet_name: str) -> pd.DataFrame:
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_file(GOOGLE_SERVICE_ACCOUNT_JSON, scopes=scopes)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id).worksheet(worksheet_name)

    # جلب البيانات كقائمة صفوف ثم تحويلها لـ DataFrame يدوياً لتجنب مشاكل تكرار العناوين
    data = sheet.get_all_values()
    if not data or len(data) <= 1:
        raise ValueError(f"The worksheet '{worksheet_name}' is empty or has no data.")

    headers = data[0]
    # معالجة الأسماء المكررة في العناوين بإضافة رقم تسلسلي تلقائياً
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
        
        # ⬇️ التعديل هنا: إضافة تنبيه صارم يمنع حذف الـ Outliers والـ 1500
        prompt = f"""You are a pandas expert. DataFrame `df` is loaded. Columns: {list(df.columns)}.
Question: "{question_description}"

Write ONLY executable Python (pandas) code assigning the answer to a variable named `result`.
- CRITICAL INSTRUCTION: Never drop, filter out, or ignore outliers, extreme values, or high values (such as 1500 or any max value) unless the user explicitly and directly asks you to remove outliers. Always include all data points (e.g., Al-Nokhba neighborhood maximum value of 1500 must be included).
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

def run_agent_turn(agent_llm, tools_by_name: dict, chat_history: list, question: str) -> str:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + chat_history + [HumanMessage(content=question)]
    for _ in range(MAX_AGENT_STEPS):
        response = agent_llm.invoke(messages)
        accumulate_token_usage(response)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            return response.content or "لم يتمكن النموذج من توليد إجابة."
        for call in tool_calls:
            selected_tool = tools_by_name.get(call["name"])
            tool_output = selected_tool.invoke(call["args"]) if selected_tool else f"Unknown tool: {call['name']}"
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=call["id"]))
    return "توقفت بعد عدة محاولات."


st.set_page_config(page_title="اسأل عن البيانات", page_icon="📊")
st.title("📊 اسأل عن بيانات الشيتات والتابات المختلفة")

if not GOOGLE_SHEET_ID or not GROQ_API_KEY or not GROQ_MODEL:
    st.error("تأكد من وجود GOOGLE_SHEET_ID و GROQ_API_KEY و GROQ_MODEL في ملف .env")
    st.stop()

render_token_sidebar()

try:
    datasets = {}

    # تحميل التابتين من الشيت الأول
    if GOOGLE_SHEET_ID and GOOGLE_WORKSHEET_NAME:
        datasets[DATASET_1_LABEL] = load_sheet_as_dataframe(GOOGLE_SHEET_ID, GOOGLE_WORKSHEET_NAME)
    if GOOGLE_SHEET_ID and GOOGLE_WORKSHEET_NAME_2:
        datasets[DATASET_2_LABEL] = load_sheet_as_dataframe(GOOGLE_SHEET_ID, GOOGLE_WORKSHEET_NAME_2)

    # تحميل التابتين من الشيت الثاني (مسار مكة)
    if GOOGLE_SHEET_ID_3 and GOOGLE_WORKSHEET_NAME_3:
        datasets[DATASET_3_LABEL] = load_sheet_as_dataframe(GOOGLE_SHEET_ID_3, GOOGLE_WORKSHEET_NAME_3)
    if GOOGLE_SHEET_ID_3 and GOOGLE_WORKSHEET_NAME_4:
        datasets[DATASET_4_LABEL] = load_sheet_as_dataframe(GOOGLE_SHEET_ID_3, GOOGLE_WORKSHEET_NAME_4)

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
            answer = run_agent_turn(agent_llm, tools_by_name, st.session_state.lc_history, question)
        st.markdown(answer)

    st.session_state.messages.append(("assistant", answer))
    st.session_state.lc_history.extend([HumanMessage(content=question), AIMessage(content=answer)])