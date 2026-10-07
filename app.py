
import json
import re
from io import BytesIO

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(page_title="DataGuard AI", page_icon="🛡️", layout="wide")

st.title("🛡️ DataGuard AI")
st.subheader("Proof-Carrying Data Analyst")
st.caption("Ask questions about your data. Get an answer, evidence, and runnable verification code.")

# -----------------------------
# Data quality
# -----------------------------
def quality_report(df):
    missing = df.isna().sum()
    missing = missing[missing > 0].to_dict()
    duplicate_count = int(df.duplicated().sum())

    numeric = df.select_dtypes(include="number")
    outliers = {}
    for col in numeric.columns:
        if numeric[col].nunique() >= 5:
            q1 = numeric[col].quantile(0.25)
            q3 = numeric[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                mask = (numeric[col] < q1 - 1.5 * iqr) | (numeric[col] > q3 + 1.5 * iqr)
                count = int(mask.sum())
                if count:
                    outliers[col] = count

    return missing, duplicate_count, outliers


# -----------------------------
# Safe question engine
# -----------------------------
def normalize(s):
    return re.sub(r"[^a-z0-9_ ]+", " ", str(s).lower()).strip()


def find_column(df, words):
    cols = list(df.columns)
    normalized = {normalize(c): c for c in cols}

    for word in words:
        for n, original in normalized.items():
            if word in n:
                return original
    return None


def local_intent(df, question):
    q = normalize(question)
    numeric_cols = list(df.select_dtypes(include="number").columns)

    # Average / mean
    if any(x in q for x in ["average", "mean"]):
        col = find_column(df, ["revenue", "sales", "amount", "price", "profit", "salary"])
        if col is None and numeric_cols:
            col = numeric_cols[0]
        if col:
            return {"operation": "mean", "column": col}

    # Total / sum
    if any(x in q for x in ["total", "sum", "overall"]):
        col = find_column(df, ["revenue", "sales", "amount", "price", "profit", "salary"])
        if col is None and numeric_cols:
            col = numeric_cols[0]
        if col:
            return {"operation": "sum", "column": col}

    # Maximum
    if any(x in q for x in ["highest", "maximum", "max", "largest", "most"]):
        value_col = find_column(df, ["revenue", "sales", "amount", "price", "profit", "salary"])
        group_col = find_column(df, ["city", "region", "category", "product", "department", "customer"])
        if group_col and value_col:
            return {"operation": "group_sum_max", "group_column": group_col, "value_column": value_col}
        if value_col:
            return {"operation": "max", "column": value_col}

    # Minimum
    if any(x in q for x in ["lowest", "minimum", "min", "smallest", "least"]):
        value_col = find_column(df, ["revenue", "sales", "amount", "price", "profit", "salary"])
        if value_col:
            return {"operation": "min", "column": value_col}

    # Count
    if any(x in q for x in ["how many", "count", "number of"]):
        group_col = find_column(df, ["city", "region", "category", "product", "department", "customer"])
        if group_col:
            return {"operation": "group_count", "group_column": group_col}
        return {"operation": "count_rows"}

    # Show columns
    if "column" in q or "columns" in q:
        return {"operation": "columns"}

    return None


def ask_llm(df, question):
    if OpenAI is None:
        return None, "OpenAI package is not installed."

    if "OPENAI_API_KEY" not in st.secrets:
        return None, "No OpenAI API key configured."

    schema = []
    for c in df.columns:
        schema.append({
            "name": str(c),
            "dtype": str(df[c].dtype)
        })

    allowed = [
        "mean", "sum", "max", "min",
        "group_sum_max", "group_count",
        "count_rows", "columns"
    ]

    prompt = f"""
You are the intent parser for a data-analysis application.

Dataset columns:
{json.dumps(schema, indent=2)}

User question:
{question}

Return ONLY valid JSON with:
operation: one of {allowed}
column: column name if needed
group_column: grouping column if needed
value_column: numeric value column if needed

Never invent a column. If the question cannot be answered from these columns,
return {{"operation":"cannot_answer","reason":"..."}}.
"""

    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        data = json.loads(response.choices[0].message.content)
        return data, None
    except Exception as e:
        return None, str(e)


def execute_intent(df, intent):
    op = intent.get("operation")

    if op == "cannot_answer":
        return None, intent.get("reason", "The question cannot be answered from this dataset."), None

    if op == "columns":
        result = list(map(str, df.columns))
        code = "result = list(df.columns)"
        return result, "The dataset contains these columns.", code

    if op == "count_rows":
        result = int(len(df))
        code = "result = len(df)"
        return result, f"The dataset contains {result} rows.", code

    if op in ["mean", "sum", "max", "min"]:
        col = intent.get("column")
        if col not in df.columns:
            return None, f"Column '{col}' is not present in the dataset.", None
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            return None, f"Column '{col}' has no usable numeric values.", None

        if op == "mean":
            value = float(series.mean())
            code = f"result = df[{col!r}].mean()"
            explanation = f"Calculated the average of '{col}' using {len(series)} numeric values."
        elif op == "sum":
            value = float(series.sum())
            code = f"result = df[{col!r}].sum()"
            explanation = f"Calculated the total of '{col}' using {len(series)} numeric values."
        elif op == "max":
            value = float(series.max())
            code = f"result = df[{col!r}].max()"
            explanation = f"Found the maximum value in '{col}'."
        else:
            value = float(series.min())
            code = f"result = df[{col!r}].min()"
            explanation = f"Found the minimum value in '{col}'."

        return value, explanation, code

    if op == "group_sum_max":
        group = intent.get("group_column")
        value = intent.get("value_column")
        if group not in df.columns or value not in df.columns:
            return None, "The requested grouping/value column was not found.", None

        temp = df[[group, value]].copy()
        temp[value] = pd.to_numeric(temp[value], errors="coerce")
        temp = temp.dropna(subset=[value])
        grouped = temp.groupby(group)[value].sum().sort_values(ascending=False)

        if grouped.empty:
            return None, "There are no usable values for this analysis.", None

        winner = grouped.index[0]
        amount = float(grouped.iloc[0])
        code = (
            f"result = (df.groupby({group!r})[{value!r}]\n"
            f"          .sum()\n"
            f"          .sort_values(ascending=False))"
        )
        return {"winner": str(winner), "value": amount, "table": grouped.head(10)}, \
               f"Grouped '{value}' by '{group}' and selected the highest total.", code

    if op == "group_count":
        group = intent.get("group_column")
        if group not in df.columns:
            return None, "The requested grouping column was not found.", None

        grouped = df.groupby(group).size().sort_values(ascending=False)
        code = (
            f"result = df.groupby({group!r}).size()"
        )
        return grouped.head(10), f"Counted rows for each '{group}'.", code

    return None, "I cannot determine a safe operation for this question.", None


# -----------------------------
# UI
# -----------------------------
with st.sidebar:
    st.header("1. Upload data")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    st.markdown("---")
    st.header("Demo questions")
    st.write("Try:")
    st.code("What is the average revenue?")
    st.code("Which city has the highest revenue?")
    st.code("What is the total sales?")
    st.code("How many customers are there?")

if uploaded is None:
    st.info("Upload a CSV file to begin.")
    st.markdown("""
### What DataGuard AI does

1. Inspects the uploaded dataset.
2. Detects data-quality problems.
3. Understands a natural-language question.
4. Performs the calculation with Python/Pandas.
5. Shows the result **and runnable verification code**.
6. Refuses questions that cannot be supported by the data.
""")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read this CSV: {e}")
    st.stop()

st.success(f"Loaded **{uploaded.name}** — {len(df):,} rows × {len(df.columns)} columns")

tab1, tab2, tab3 = st.tabs(["🔎 Ask DataGuard", "🧹 Data Quality", "📋 Raw Data"])

with tab1:
    question = st.text_input(
        "Ask a question about your dataset",
        placeholder="Example: Which city has the highest revenue?"
    )

    if st.button("🚀 Analyze", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analyzing..."):
                intent = local_intent(df, question)
                source = "Built-in safe intent engine"

                if intent is None:
                    intent, err = ask_llm(df, question)
                    source = "LLM intent parser"
                    if intent is None:
                        st.warning("I couldn't safely understand that question.")
                        st.info("Try one of the demo questions, or configure an OpenAI API key for natural-language questions.")
                        if err:
                            st.caption(f"Details: {err}")
                        st.stop()

                result, explanation, code = execute_intent(df, intent)

            if result is None:
                st.error(explanation)
            else:
                st.success("🟢 VERIFIED ANSWER")
                st.caption(f"Reasoning source: {source}")

                if isinstance(result, dict) and "winner" in result:
                    st.metric("Top group", result["winner"])
                    st.metric("Value", f"{result['value']:,.2f}")
                    st.dataframe(result["table"], use_container_width=True)
                    st.bar_chart(result["table"])
                elif isinstance(result, pd.Series):
                    st.dataframe(result, use_container_width=True)
                    st.bar_chart(result)
                elif isinstance(result, list):
                    st.write(result)
                elif isinstance(result, (int, float)):
                    st.metric("Answer", f"{result:,.2f}")
                else:
                    st.write(result)

                st.write("### 🔍 Evidence")
                st.write(explanation)

                st.write("### 💻 Verification code")
                st.code(code, language="python")

                st.info(
                    "The displayed result is produced by executing the corresponding "
                    "Pandas operation on the uploaded dataset."
                )

with tab2:
    missing, duplicates, outliers = quality_report(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns):,}")
    c3.metric("Duplicate rows", f"{duplicates:,}")

    st.write("### Missing values")
    if missing:
        st.dataframe(pd.DataFrame(
            {"Column": list(missing.keys()), "Missing values": list(missing.values())}
        ), use_container_width=True)
    else:
        st.success("No missing values detected.")

    st.write("### Possible numeric outliers")
    if outliers:
        st.dataframe(pd.DataFrame(
            {"Column": list(outliers.keys()), "Possible outliers": list(outliers.values())}
        ), use_container_width=True)
    else:
        st.success("No obvious IQR-based outliers detected.")

    st.write("### Data types")
    st.dataframe(
        pd.DataFrame({"Column": df.columns.astype(str), "Type": df.dtypes.astype(str)}),
        use_container_width=True
    )

with tab3:
    st.dataframe(df, use_container_width=True)
