# Databricks notebook source
import os
import re
import json
import time
import hashlib
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple

import pandas as pd
from openai import OpenAI
from pyspark.sql import functions as F, types as T
from pyspark.sql.window import Window

SOURCE_TABLE = "`prod-sandbox`.vivekkumar_patel.gold_tnet_tripmaster"
CACHE_TABLE = "`prod-sandbox`.vivekkumar_patel.tnet_clinical_qa_llm_cache"
OUTPUT_TABLE = "`prod-sandbox`.vivekkumar_patel.tnet_clinical_qa"
WRITE_OUTPUT = True
JSON_OUTPUT_DIR = os.path.join(os.getcwd(), "json_payloads")

WORKSPACE_BASE_URL = "https://adb-2790612761746757.17.azuredatabricks.net/serving-endpoints"
LLM_MODEL = "databricks-gpt-oss-120b"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 4000
LLM_RETRIES = 3
MAX_WORKERS = 8
FLUSH_EVERY = 200
PROMPT_VERSION = "qa_v2"

# COMMAND ----------

DATABRICKS_TOKEN = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    if "dbutils" in dir() else os.environ.get("DATABRICKS_TOKEN", "")
)

client = OpenAI(api_key=DATABRICKS_TOKEN, base_url=WORKSPACE_BASE_URL)


def llm_call(system_prompt: str, messages: List[Dict[str, str]],
             max_tokens: int = LLM_MAX_TOKENS) -> str:
    payload = [{"role": "system", "content": system_prompt}] + messages
    last_error = None
    for attempt in range(LLM_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=payload,
                temperature=LLM_TEMPERATURE,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        return item.get("text", "")
                return json.dumps(content)
            return content
        except Exception as e:
            last_error = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"model call failed after {LLM_RETRIES} attempts: {last_error}")

# COMMAND ----------

SYSTEM_PROMPT = """You extract question and answer pairs from ambulance transport order form text.

The text is a web form flattened into one string. Its characteristics:
- A question label is often printed twice in a row, and the second copy may end with "?:" or ":".
- Instruction text is neither a question nor an answer. Examples: "(One selection required)", "One selection required", "(Select all that apply)", "Select all that apply", "(Select one)", "(Yes/No)", "At least one selection required", "(Please specify)".
- A question is any prompt or field label on the form, whether or not it ends in "?". Statement labels such as "Number of Drips", "Patient stay status", "Select unit assignment", or "Transport Reason" are questions.
- The answer is the value that was entered or selected. It appears after the question label and before the next question label.
- A selected option can itself be a label with its own value, for example "The patient requires continuous oxygen and cannot self-monitor: 2LNC". Report that as its own question with its own value.
- Several selected options in a row form one answer. Copy the whole contiguous span.

RULES
- An answer must come immediately after its question label, with only instruction text or a repeat of the label in between.
- If a question is followed by instruction text such as "One selection required:" or "(Select all that apply)" and the next thing after that is another question, the first question was never answered. Omit it. Never pair it with an answer that appears later in the text.
- Omit any question followed directly by another question, by a repeat of itself, by instruction text only, or by the end of the text.
- Copy answers verbatim as a contiguous span of the source text. Do not paraphrase, correct, reorder, or infer.
- Copy questions verbatim, without trailing "?" or ":" and without repeating the label twice.
- Report each question once.
- If nothing is answered, return {"qa": []}.

OUTPUT: valid JSON only. No prose, no markdown fences.
{"qa": [{"question": string, "answer": string}]}"""

FEWSHOT_PAIRS = [
    (
        "Is this an ED Transport? Is this an ED Transport? What is the reason for patient transport? One selection required: Is this a round trip transport? (One selection required) Is this a round trip transport?",
        {"qa": []},
    ),
    (
        "Is this an ED Transport? Is this an ED Transport?: No What is the reason for patient transport? One selection required: Discharge to lower level of care or home. Is any special handling required during transport? (Select all that apply) The patient requires special positioning or handling due to illness, injury, recent surgery, contractures, orthopedic device; or other reason (Please specify): wheelchair bound but do not have wheelchair. Is this a round trip transport? (One selection required) Is this a round trip transport?: No. A return trip is not required.",
        {"qa": [
            {"question": "Is this an ED Transport", "answer": "No"},
            {"question": "What is the reason for patient transport", "answer": "Discharge to lower level of care or home."},
            {"question": "The patient requires special positioning or handling due to illness, injury, recent surgery, contractures, orthopedic device; or other reason", "answer": "wheelchair bound but do not have wheelchair."},
            {"question": "Is this a round trip transport", "answer": "No. A return trip is not required."},
        ]},
    ),
    (
        "Specify what services are required: (Select all that apply) Cardiac services are not available Patient stay status Is the patient's stay covered under Medicare Part A (PPS/DRG?) (One selection required) No This is a Time Sensitive transport request (Select one) Due to Patient Clinical condition The patient requires the following special handling, medical care, monitoring or equipment during transport: (Select all that apply) The patient requires continuous oxygen and cannot self-monitor: 2LNC Transport Requirements Select all that apply Number of Drips: 2",
        {"qa": [
            {"question": "Specify what services are required", "answer": "Cardiac services are not available"},
            {"question": "Is the patient's stay covered under Medicare Part A (PPS/DRG?)", "answer": "No"},
            {"question": "This is a Time Sensitive transport request", "answer": "Due to Patient Clinical condition"},
            {"question": "The patient requires continuous oxygen and cannot self-monitor", "answer": "2LNC"},
            {"question": "Number of Drips", "answer": "2"},
        ]},
    ),
]

FEWSHOT = []
for text, out in FEWSHOT_PAIRS:
    FEWSHOT.append({"role": "user", "content": text})
    FEWSHOT.append({"role": "assistant", "content": json.dumps(out)})

# COMMAND ----------

INSTRUCTION_ONLY_RE = re.compile(
    r"^[\s(]*(one\s+sel\w*\s+required|select\s+all\s+that\s+apply|select\s+one|yes\s*/\s*no|"
    r"at\s+least\s+one\s+sel\w*\s+required|please\s+specify)[\s).:]*$",
    re.IGNORECASE,
)


def normalize_text(s: str) -> str:
    s = re.sub(r"(?<=[A-Za-z])\?s\b", "'s", str(s))
    return re.sub(r"\s+", " ", s).strip().lower()


INSTRUCTION_PREFIX_RE = re.compile(
    r"^(?:[\s:?.\-()]+|one\s+sel\w*\s+required|select\s+all\s+that\s+apply|select\s+one|yes\s*/\s*no|"
    r"at\s+least\s+one\s+sel\w*\s+required|please\s+specify)",
    re.IGNORECASE,
)


def answer_follows_question(q: str, a: str, src: str) -> bool:
    qn, an = normalize_text(q), normalize_text(a)
    start = src.find(qn)
    while start != -1:
        rest = src[start + len(qn):]
        while True:
            m = INSTRUCTION_PREFIX_RE.match(rest)
            if m and m.end() > 0:
                rest = rest[m.end():]
            elif rest.startswith(qn):
                rest = rest[len(qn):]
            else:
                break
        if rest.startswith(an):
            return True
        start = src.find(qn, start + 1)
    return False


def strip_fences(raw: str) -> str:
    txt = str(raw).strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-zA-Z]*\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt)
    start, end = txt.find("{"), txt.rfind("}")
    return txt[start:end + 1] if start != -1 and end > start else txt


def validate_response(raw: str, source_text: str) -> Tuple[bool, List[Dict[str, str]], int, str]:
    try:
        obj = json.loads(strip_fences(raw))
    except Exception as e:
        return False, [], 0, f"invalid JSON: {e}"
    items = obj.get("qa") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return False, [], 0, "missing qa array"

    src = normalize_text(source_text)
    kept, seen, dropped = [], set(), 0
    for it in items:
        if not isinstance(it, dict):
            dropped += 1
            continue
        q = re.sub(r"\s+", " ", str(it.get("question", ""))).strip().rstrip("?:. ").strip()
        a = re.sub(r"\s+", " ", str(it.get("answer", ""))).strip().lstrip(":?. ").strip()
        if not q or not a or not re.search(r"[A-Za-z0-9]", a):
            dropped += 1
            continue
        if INSTRUCTION_ONLY_RE.match(a) or normalize_text(a) == normalize_text(q):
            dropped += 1
            continue
        if not answer_follows_question(q, a, src):
            dropped += 1
            continue
        key = normalize_text(q)
        if key in seen:
            continue
        seen.add(key)
        kept.append({"question": q, "answer": a})
    return True, kept, dropped, ""


def extract_one(text_hash: str, text: str) -> Dict[str, Any]:
    row = {
        "text_hash": text_hash,
        "prompt_version": PROMPT_VERSION,
        "status": "ok",
        "qa_json": "[]",
        "qa_count": 0,
        "dropped_ungrounded": 0,
        "error": None,
        "processed_at": datetime.datetime.utcnow(),
    }
    if not text or not text.strip():
        return row
    try:
        raw = llm_call(SYSTEM_PROMPT, FEWSHOT + [{"role": "user", "content": text}])
        ok, qa, dropped, err = validate_response(raw, text)
        if not ok:
            raw = llm_call(SYSTEM_PROMPT, FEWSHOT + [
                {"role": "user", "content": text},
                {"role": "assistant", "content": raw or ""},
                {"role": "user", "content": f"That response was rejected ({err}). Return only valid JSON matching the schema."},
            ])
            ok, qa, dropped, err = validate_response(raw, text)
        if not ok:
            row.update(status="error", error=err)
            return row
        row.update(qa_json=json.dumps(qa), qa_count=len(qa), dropped_ungrounded=dropped)
    except Exception as e:
        row.update(status="error", error=str(e)[:1000])
    return row

# COMMAND ----------

src = (spark.sql(f"""
        SELECT TripRequestId, RequestDate, ClinicalData
        FROM {SOURCE_TABLE}
        WHERE ClinicalData IS NOT NULL AND trim(ClinicalData) <> ''
    """)
    .withColumn("text_hash", F.sha2(F.col("ClinicalData"), 256)))

distinct_texts = src.select("text_hash", "ClinicalData").dropDuplicates(["text_hash"])

CACHE_SCHEMA = T.StructType([
    T.StructField("text_hash", T.StringType()),
    T.StructField("prompt_version", T.StringType()),
    T.StructField("status", T.StringType()),
    T.StructField("qa_json", T.StringType()),
    T.StructField("qa_count", T.IntegerType()),
    T.StructField("dropped_ungrounded", T.IntegerType()),
    T.StructField("error", T.StringType()),
    T.StructField("processed_at", T.TimestampType()),
])

if not spark.catalog.tableExists(CACHE_TABLE.replace("`", "")):
    spark.createDataFrame([], CACHE_SCHEMA).write.saveAsTable(CACHE_TABLE)

done = (spark.table(CACHE_TABLE)
    .where((F.col("prompt_version") == PROMPT_VERSION) & (F.col("status") == "ok"))
    .select("text_hash").distinct())

todo_pdf = distinct_texts.join(done, "text_hash", "left_anti").toPandas()

print(f"Source rows          : {src.count():,}")
print(f"Distinct texts       : {distinct_texts.count():,}")
print(f"Already processed    : {done.count():,}")
print(f"To send to the model : {len(todo_pdf):,}")

# COMMAND ----------

def flush(buffer: List[Dict[str, Any]]) -> None:
    if buffer:
        spark.createDataFrame(pd.DataFrame(buffer), CACHE_SCHEMA).write.mode("append").saveAsTable(CACHE_TABLE)


buffer, completed, errors = [], 0, 0
started = time.time()
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = [pool.submit(extract_one, r.text_hash, r.ClinicalData) for r in todo_pdf.itertuples()]
    for fut in as_completed(futures):
        res = fut.result()
        buffer.append(res)
        completed += 1
        errors += res["status"] == "error"
        if len(buffer) >= FLUSH_EVERY:
            flush(buffer)
            buffer = []
            print(f"{completed:,}/{len(futures):,} done | errors {errors:,} | {time.time() - started:,.0f}s")
flush(buffer)
print(f"{completed:,}/{len(todo_pdf):,} done | errors {errors:,} | {time.time() - started:,.0f}s")

# COMMAND ----------

QA_SCHEMA = T.ArrayType(T.StructType([
    T.StructField("question", T.StringType()),
    T.StructField("answer", T.StringType()),
]))

latest = (spark.table(CACHE_TABLE)
    .where(F.col("prompt_version") == PROMPT_VERSION)
    .withColumn("rn", F.row_number().over(
        Window.partitionBy("text_hash").orderBy(F.desc(F.col("status") == "ok"), F.desc("processed_at"))))
    .where("rn = 1")
    .drop("rn"))

parsed = (src.join(latest, "text_hash", "left")
    .withColumn("qa", F.from_json("qa_json", QA_SCHEMA)))

qa_by_trip = (parsed
    .where((F.col("status") == "ok") & (F.col("qa_count") > 0))
    .groupBy("TripRequestId")
    .agg(F.max("RequestDate").alias("RequestDate"),
         F.array_distinct(F.flatten(F.collect_list("qa"))).alias("qa"))
    .withColumn("qa_count", F.size("qa"))
    .withColumn("qa_json", F.to_json(F.struct("TripRequestId", "RequestDate", "qa"))))

display(parsed.groupBy(
        F.coalesce(F.col("status"), F.lit("not_processed")).alias("status"),
        (F.col("qa_count") > 0).alias("has_answers"))
    .count())
display(qa_by_trip.orderBy(F.desc("RequestDate")))

# COMMAND ----------

display(parsed.where("status = 'error'").select("TripRequestId", "ClinicalData", "error"))
display(parsed.where("dropped_ungrounded > 0")
        .select("TripRequestId", "ClinicalData", "qa", "dropped_ungrounded")
        .orderBy(F.desc("dropped_ungrounded")))
display(parsed.select(F.explode("qa.question").alias("question"))
        .groupBy("question").count().orderBy(F.desc("count")))

# COMMAND ----------

pdf = qa_by_trip.select("TripRequestId", "RequestDate", "qa_count", "qa_json").toPandas()

os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
for r in pdf.itertuples():
    with open(os.path.join(JSON_OUTPUT_DIR, f"{r.TripRequestId}.json"), "w") as f:
        json.dump(json.loads(r.qa_json), f, indent=2)
print(f"Wrote {len(pdf):,} files to {JSON_OUTPUT_DIR}")
pdf.head()

# COMMAND ----------

if WRITE_OUTPUT:
    (qa_by_trip.select("TripRequestId", "RequestDate", "qa", "qa_count", "qa_json")
        .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(OUTPUT_TABLE))
    print(f"Wrote {OUTPUT_TABLE}")
