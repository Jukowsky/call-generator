import os
import json
import uuid
import asyncio
import random
import platform
import datetime
from io import BytesIO

import requests
import urllib3
from flask import Flask, request, jsonify, send_file, send_from_directory, Response, stream_with_context
from openai import OpenAI
import edge_tts
from pydub import AudioSegment
from pydub.effects import pan as pydub_pan

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = Flask(__name__, static_folder="static", static_url_path="")

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")

os.makedirs(DATA_DIR, exist_ok=True)

conversations_store: dict = {}

VOICES = {
    "en": {"agent": "en-US-AriaNeural", "customer": "en-US-GuyNeural"},
    "tr": {"agent": "tr-TR-EmelNeural", "customer": "tr-TR-AhmetNeural"},
}

LENGTH_TURNS = {
    "short":    "6–8",
    "moderate": "12–16",
    "long":     "22–28",
}

DOMAINS = ["banking", "ecommerce", "telco", "insurance", "healthcare", "utilities"]

CATEGORY_DESCRIPTIONS = {
    "complaint":   "the customer is calling to complain about a frustrating experience or problem they encountered",
    "churn":       "the customer wants to cancel their account or service; the agent tries to understand their reasons and retain them with offers or solutions",
    "track_order": "the customer is calling to check the status of their order, delivery, or service request",
    "billing":     "the customer has a billing dispute, an unexpected charge, or a payment-related question",
    "technical":   "the customer is experiencing a technical issue and needs troubleshooting assistance",
}

DOMAIN_CONTEXT = {
    "banking":    "bank or financial institution — use realistic banking terms: account number, balance, transaction, credit card, loan, branch, interest rate",
    "ecommerce":  "online retail store — use realistic ecommerce terms: order number, delivery, return, refund, product SKU, warehouse, courier",
    "telco":      "telecommunications provider — use realistic telco terms: phone plan, data limit, network outage, monthly bill, SIM, contract, roaming",
    "insurance":  "insurance company — use realistic insurance terms: policy number, claim, premium, coverage, deductible, adjuster, renewal",
    "healthcare": "healthcare provider or clinic — use realistic healthcare terms: patient ID, appointment, prescription, referral, medical records, copay",
    "utilities":  "utilities company (electricity/gas/water) — use realistic utility terms: meter reading, outage, service address, bill, consumption, connection fee",
}

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

_settings = {
    "api_key":       "",
    "identity_url":  "",
    "base_url":      "",
    "client_id":     "",
    "client_secret": "",
    "source_name":   "Default Api Source",
    "agent_ids":     "",   # comma/newline separated UUIDs; empty = fetch from API
}


def _load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                _settings.update(json.load(f))
        except Exception as exc:
            print(f"[warn] Could not load settings: {exc}")


def _save_settings():
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(_settings, f, indent=2)
    except Exception as exc:
        print(f"[warn] Could not save settings: {exc}")


_load_settings()


def _get_client():
    key = _settings["api_key"] or os.environ.get("OPENAI_API_KEY", "")
    return OpenAI(api_key=key) if key else None


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _date_str(conv: dict) -> str:
    return (conv.get("created_at") or datetime.date.today().isoformat())[:10]


def _transcript_path(conv: dict) -> str:
    d = os.path.join(DATA_DIR, _date_str(conv), "transcripts")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{conv['id']}.json")


def _audio_path(conv: dict) -> str:
    d = os.path.join(DATA_DIR, _date_str(conv), "audio")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{conv['id']}.mp3")


def _audio_path_by_id(conv_id: str) -> str:
    conv = conversations_store.get(conv_id)
    if conv:
        return _audio_path(conv)
    for date_folder in os.listdir(DATA_DIR):
        p = os.path.join(DATA_DIR, date_folder, "audio", f"{conv_id}.mp3")
        if os.path.exists(p):
            return p
    return os.path.join(DATA_DIR, f"{conv_id}.mp3")


def _save_conversation(conv: dict):
    try:
        with open(_transcript_path(conv), "w", encoding="utf-8") as f:
            json.dump(conv, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[warn] Could not save conversation {conv.get('id')}: {exc}")


def _load_conversations():
    loaded = 0
    for date_folder in sorted(os.listdir(DATA_DIR)):
        tx_dir = os.path.join(DATA_DIR, date_folder, "transcripts")
        if not os.path.isdir(tx_dir):
            continue
        for fname in os.listdir(tx_dir):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(tx_dir, fname), "r", encoding="utf-8") as f:
                    conv = json.load(f)
                audio = os.path.join(DATA_DIR, date_folder, "audio", f"{conv['id']}.mp3")
                conv["audio_generated"] = os.path.exists(audio)
                conversations_store[conv["id"]] = conv
                loaded += 1
            except Exception as exc:
                print(f"[warn] Skipping {fname}: {exc}")
    if loaded:
        print(f"[info] Loaded {loaded} conversation(s) from disk.")


_load_conversations()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/settings", methods=["GET", "POST"])
def manage_settings():
    if request.method == "POST":
        data = request.get_json() or {}
        for key in ("api_key", "identity_url", "base_url", "client_id", "client_secret", "source_name", "agent_ids"):
            if key in data:
                _settings[key] = data[key].strip()
        _save_settings()
        return jsonify({"ok": True})
    return jsonify({
        "api_key_set":       bool(_settings["api_key"] or os.environ.get("OPENAI_API_KEY", "")),
        "identity_url":      _settings["identity_url"],
        "base_url":          _settings["base_url"],
        "client_id":         _settings["client_id"],
        "source_name":       _settings["source_name"],
        "agent_ids":         _settings["agent_ids"],
        "client_secret_set": bool(_settings["client_secret"]),
    })


@app.route("/api/conversations")
def list_conversations():
    convs = list(conversations_store.values())
    convs.sort(key=lambda c: c.get("created_at", ""), reverse=True)
    return jsonify(convs)


@app.route("/api/generate", methods=["POST"])
def generate():
    data       = request.get_json()
    domain     = data.get("domain", "random")
    categories = data.get("categories") or ["complaint"]
    count      = min(int(data.get("count", 3)), 20)
    language   = data.get("language", "en")
    length     = data.get("length", "moderate")
    negative   = bool(data.get("negative", False))

    client = _get_client()
    if not client:
        return jsonify({"error": "OpenAI API key is not set."}), 401

    @stream_with_context
    def event_stream():
        print(f"[debug] /api/generate: domain={domain} cats={categories} count={count} lang={language} length={length} negative={negative}")
        yield f"data: {json.dumps({'type': 'connected', 'total': count})}\n\n"

        for i in range(count):
            actual_domain   = domain if domain != "random" else random.choice(DOMAINS)
            actual_category = random.choice(categories)

            print(f"[debug] Generating {i+1}/{count}: {actual_domain}/{actual_category}")
            yield f"data: {json.dumps({'type': 'processing', 'current': i+1, 'total': count, 'domain': actual_domain, 'category': actual_category})}\n\n"

            try:
                conv = _generate_conversation(actual_domain, actual_category, client, language, length, negative)
                conv["id"]              = str(uuid.uuid4())
                conv["audio_generated"] = False
                conv["language"]        = language
                conv["length"]          = length
                conv["negative"]        = negative
                conv["created_at"]      = datetime.datetime.now(datetime.timezone.utc).isoformat()
                conversations_store[conv["id"]] = conv
                _save_conversation(conv)
                print(f"[debug] OK: {conv['id']}")
                yield f"data: {json.dumps({'type': 'conversation', 'current': i+1, 'total': count, 'data': conv})}\n\n"
            except Exception as exc:
                import traceback
                print(f"[debug] ERROR on {i+1}/{count}: {exc}")
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc), 'current': i+1, 'total': count})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'total': count})}\n\n"

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/synthesize/<conv_id>", methods=["POST"])
def synthesize(conv_id):
    if conv_id not in conversations_store:
        return jsonify({"error": "Conversation not found"}), 404
    conv = conversations_store[conv_id]
    try:
        audio_bytes = asyncio.run(_build_stereo_audio(conv["turns"], conv.get("language", "en")))
        path = _audio_path(conv)
        with open(path, "wb") as f:
            f.write(audio_bytes)
        conversations_store[conv_id]["audio_generated"] = True
        _save_conversation(conversations_store[conv_id])
        return jsonify({"success": True})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/audio/<conv_id>")
def stream_audio(conv_id):
    path = _audio_path_by_id(conv_id)
    if not os.path.exists(path):
        return jsonify({"error": "Audio not found"}), 404
    return send_file(path, mimetype="audio/mpeg")


@app.route("/api/download/audio/<conv_id>")
def download_audio(conv_id):
    path = _audio_path_by_id(conv_id)
    if not os.path.exists(path):
        return jsonify({"error": "Audio not found"}), 404
    conv     = conversations_store.get(conv_id, {})
    filename = f"{conv.get('domain', 'call')}_{conv.get('category', 'conv')}_{conv_id[:8]}.mp3"
    return send_file(path, mimetype="audio/mpeg", as_attachment=True, download_name=filename)


@app.route("/api/download/transcript/<conv_id>")
def download_transcript(conv_id):
    if conv_id not in conversations_store:
        return jsonify({"error": "Conversation not found"}), 404
    conv     = conversations_store[conv_id]
    text     = _format_transcript(conv)
    filename = f"{conv.get('domain', 'call')}_{conv.get('category', 'conv')}_{conv_id[:8]}.txt"
    return send_file(
        BytesIO(text.encode("utf-8")),
        mimetype="text/plain",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/api/delete/<conv_id>", methods=["DELETE"])
def delete_conversation(conv_id):
    conv = conversations_store.pop(conv_id, None)
    if conv:
        for p in (_transcript_path(conv), _audio_path(conv)):
            if os.path.exists(p):
                os.remove(p)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Knovvu push
# ---------------------------------------------------------------------------

def _normalize_identity_url(url: str) -> str:
    url = url.rstrip("/")
    for suffix in ("/connect/token", "/connect"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url


def _normalize_base_url(url: str) -> str:
    url = url.rstrip("/")
    for suffix in ("/ext-api/v1", "/ext-api"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url


def _get_knovvu_token() -> str:
    identity_url  = _normalize_identity_url(_settings["identity_url"])
    client_id     = _settings["client_id"]
    client_secret = _settings["client_secret"]
    if not (identity_url and client_id and client_secret):
        raise ValueError("Identity URL, Client ID and Client Secret must all be configured")
    resp = requests.post(
        f"{identity_url}/connect/token",
        data={
            "grant_type":    "client_credentials",
            "client_id":     client_id,
            "client_secret": client_secret,
            "scope":         "CAExternal",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=10,
        verify=False,
    )
    resp.raise_for_status()
    token = resp.json().get("access_token")
    if not token:
        raise ValueError("No access_token in identity server response")
    print("[debug] Obtained Knovvu token via client_credentials")
    return token


@app.route("/api/push/<conv_id>", methods=["POST"])
def push_conversation(conv_id):
    if conv_id not in conversations_store:
        return jsonify({"error": "Conversation not found"}), 404

    base_url = _normalize_base_url(_settings["base_url"])
    if not base_url:
        return jsonify({"error": "Knovvu Base URL not configured"}), 400
    if not (_settings["client_id"] and _settings["client_secret"]):
        return jsonify({"error": "Client ID and Client Secret must be configured"}), 400

    try:
        step = "get token"
        token   = _get_knovvu_token()
        headers = {"Authorization": f"Bearer {token}", "accept": "text/plain"}

        # 1. Resolve agent ID
        step = "resolve agent"
        pinned = [i.strip() for i in _settings["agent_ids"].replace("\n", ",").split(",") if i.strip()]
        if pinned:
            agent_id = random.choice(pinned)
            print(f"[debug] Using pinned agent_id={agent_id} ({len(pinned)} in pool)")
        else:
            step = "get users"
            users_resp = requests.get(
                f"{base_url}/ext-api/v1/users",
                params={"OnlyActiveUsers": "true", "SkipCount": 0, "MaxCount": 500},
                headers=headers,
                timeout=15,
                verify=False,
            )
            print(f"[debug] GET /users → {users_resp.status_code}: {users_resp.text[:300]}")
            users_resp.raise_for_status()
            raw = users_resp.json()
            if isinstance(raw, dict):
                users = raw.get("users") or raw.get("items") or []
            else:
                users = raw if isinstance(raw, list) else []
            if not users:
                return jsonify({"error": f"No users returned by Knovvu. Response: {str(raw)[:200]}"}), 400

            def _has_role(u, role_name):
                return any(r.get("name") == role_name for r in u.get("roles", []))

            agent_users = [u for u in users if u.get("isActive") and u.get("isValid") and _has_role(u, "Agent")]
            pool     = agent_users or users
            user     = random.choice(pool)
            agent_id = str(user.get("id") or "")
            print(f"[debug] Selected agent '{user.get('fullName')}' (id={agent_id}) from {len(pool)} eligible agents")

        conv = conversations_store[conv_id]

        # 2. Synthesize audio if not already done
        step = "synthesize audio"
        audio_path = _audio_path(conv)
        if not os.path.exists(audio_path):
            print(f"[debug] Synthesizing audio for {conv_id} before push")
            audio_bytes = asyncio.run(_build_stereo_audio(conv["turns"], conv.get("language", "en")))
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
            conv["audio_generated"] = True
            conversations_store[conv_id]["audio_generated"] = True
            _save_conversation(conv)

        lang      = conv.get("language", "en")
        lang_code = "tr-TR" if lang == "tr" else "en-US"
        now       = datetime.datetime.now(datetime.timezone.utc)

        turn_count   = len(conv.get("turns", []))
        duration_sec = max(turn_count * 15, 60)
        end_time     = now.isoformat().replace("+00:00", "Z")
        start_time   = (now - datetime.timedelta(seconds=duration_sec)).isoformat().replace("+00:00", "Z")

        # 3. POST call
        step = "post call"
        call_body = {
            "externalId":        str(uuid.uuid4()),
            "directionId":       0,
            "duration":          duration_sec,
            "callerNumber":      "5551234567",
            "calledNumber":      "5559876543",
            "releasingPartyId":  0,
            "ratingForNPS":      0,
            "uniqueId":          conv_id,
            "relatedGroupId":    None,
            "uniqueCustomerKey": conv.get("customer_name", "unknown"),
            "agent": {
                "pbxAgentId": "",
                "deviceId":   "",
                "externalId": "",
                "agentId":    agent_id,
            },
            "attachedData": [
                {"key": "domain",    "value": conv.get("domain",   "")},
                {"key": "category",  "value": conv.get("category", "")},
            ],
            "startTime":          start_time,
            "endTime":            end_time,
            "sourceName":         _settings["source_name"] or "Default Api Source",
            "languageCode":       lang_code,
            "agentChannelId":     1,
            "hasScreenRecording": False,
            "ratingForCSAT":      0,
        }

        call_resp = requests.post(
            f"{base_url}/ext-api/v1/calls",
            json=call_body,
            headers={**headers, "Content-Type": "application/json"},
            timeout=15,
            verify=False,
        )
        print(f"[debug] POST /calls → {call_resp.status_code}: {call_resp.text[:500]}")
        call_resp.raise_for_status()

        # API returns { externalId, conversationId, ... }
        step = "parse call response"
        call_data   = call_resp.json()
        external_id = None
        conv_id_api = None
        if isinstance(call_data, dict):
            external_id = call_data.get("externalId")
            conv_id_api = call_data.get("conversationId") or call_data.get("id") or call_data.get("callId")
        elif isinstance(call_data, (int, float)):
            conv_id_api = int(call_data)

        if not external_id and not conv_id_api:
            return jsonify({"error": f"Could not extract call ID from response: {str(call_data)[:200]}"}), 500

        audio_key = external_id or conv_id_api
        print(f"[debug] conversationId={conv_id_api}  externalId={external_id}  using key={audio_key} for audio PUT")

        # 4. PUT audio — API expects externalId in the URL
        step = "upload audio"
        with open(audio_path, "rb") as f:
            audio_resp = requests.put(
                f"{base_url}/ext-api/v1/calls/{audio_key}/audio",
                files={"file": (f"{conv_id}.mp3", f, "audio/mpeg")},
                headers=headers,
                timeout=120,
                verify=False,
            )
        print(f"[debug] PUT /calls/{audio_key}/audio → {audio_resp.status_code}: {audio_resp.text[:300]}")
        audio_resp.raise_for_status()

        conv["pushed"]       = True
        conv["push_call_id"] = str(conv_id_api or audio_key)
        conversations_store[conv_id] = conv
        _save_conversation(conv)

        return jsonify({"success": True, "call_id": conv_id_api or audio_key})

    except requests.HTTPError as exc:
        body = exc.response.text[:400] if exc.response is not None else "(no response)"
        code = exc.response.status_code if exc.response is not None else "?"
        msg  = f"[{step}] HTTP {code}: {body}"
        print(f"[debug] Push HTTP error: {msg}")
        return jsonify({"error": msg}), 500
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"[{step}] {type(exc).__name__}: {exc}"}), 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_conversation(domain, category, client, language="en", length="moderate", negative=False) -> dict:
    cat_desc    = CATEGORY_DESCRIPTIONS.get(category, f"customer has a {category.replace('_', ' ')} issue")
    dom_ctx     = DOMAIN_CONTEXT.get(domain, domain)
    turns_range = LENGTH_TURNS.get(length, "12–16")
    lang_instruction = (
        "The entire conversation must be in Turkish (Türkçe). Use Turkish names and a Turkish company name."
        if language == "tr" else
        "The entire conversation must be in English."
    )
    sentiment_instruction = (
        "Sentiment: The customer is clearly frustrated and emotionally negative throughout — they may raise their voice, "
        "use sharp or impatient language, express distrust, threaten to escalate or leave, and resist the agent's solutions. "
        "The agent stays professional but the call ends with the issue only partially resolved or the customer still dissatisfied."
        if negative else
        "Sentiment: Keep the overall tone neutral to positive — the customer is firm but polite, and the agent successfully resolves the issue."
    )

    prompt = f"""Generate a realistic, natural-sounding call center conversation for a {dom_ctx}.
Scenario: {cat_desc}.
{lang_instruction}
{sentiment_instruction}

Guidelines:
- Agent greets the caller, states the company name and their own first name
- Agent verifies customer identity (full name + account/order/policy number)
- Customer describes their specific issue with realistic details
- Agent shows empathy, follows proper procedure, asks clarifying questions
- Realistic back-and-forth dialogue — not scripted or robotic
- Agent reaches a resolution or clearly explains next steps
- Professional, warm closing
- Total dialogue turns: {turns_range}

Respond ONLY with valid JSON — no markdown fences, no extra text:
{{"domain":"{domain}","category":"{category}","customer_name":"FirstName LastName","account_ref":"realistic reference number","agent_name":"FirstName","company_name":"realistic company name for {domain}","summary":"one-sentence summary of the call","turns":[{{"speaker":"agent","text":"..."}},{{"speaker":"customer","text":"..."}}]}}"""

    print(f"[debug] Calling OpenAI gpt-4o ...")
    msg  = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=3500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.choices[0].message.content.strip()
    print(f"[debug] Raw response (first 200 chars): {text[:200]}")

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    return json.loads(text)


async def _build_stereo_audio(turns: list, language: str = "en") -> bytes:
    lang_voices = VOICES.get(language, VOICES["en"])
    segments    = []

    for turn in turns:
        voice     = lang_voices["agent"] if turn["speaker"] == "agent" else lang_voices["customer"]
        mp3_bytes = await _tts(turn["text"], voice)
        # Ensure stereo BEFORE panning — panning a mono segment produces identical L/R channels
        segment   = AudioSegment.from_mp3(BytesIO(mp3_bytes)).set_channels(2)
        pan_val   = -1.0 if turn["speaker"] == "agent" else 1.0   # hard left / hard right
        segment   = pydub_pan(segment, pan_val)
        segments.append(segment)

    if not segments:
        return b""

    silence  = AudioSegment.silent(duration=350,
                                   frame_rate=segments[0].frame_rate).set_channels(2)
    combined = segments[0]
    for seg in segments[1:]:
        combined = combined + silence + seg

    out = BytesIO()
    combined.export(out, format="mp3", bitrate="128k")
    return out.getvalue()


async def _tts(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)


def _format_transcript(conv: dict) -> str:
    lang_label = "Turkish" if conv.get("language") == "tr" else "English"
    sep   = "=" * 62
    lines = [
        sep,
        "  CALL CENTER CONVERSATION TRANSCRIPT",
        sep,
        f"  Domain    : {conv.get('domain', '').upper()}",
        f"  Category  : {conv.get('category', '').replace('_', ' ').title()}",
        f"  Language  : {lang_label}",
        f"  Length    : {conv.get('length', 'moderate').title()}",
        f"  Company   : {conv.get('company_name', 'N/A')}",
        f"  Agent     : {conv.get('agent_name', 'N/A')}",
        f"  Customer  : {conv.get('customer_name', 'N/A')}",
        f"  Reference : {conv.get('account_ref', 'N/A')}",
        f"  Summary   : {conv.get('summary', 'N/A')}",
        sep,
        "",
    ]
    for turn in conv.get("turns", []):
        label = f"AGENT — {conv.get('agent_name', 'Agent')}" if turn["speaker"] == "agent" \
                else f"CUSTOMER — {conv.get('customer_name', 'Customer')}"
        lines.append(f"[{label}]")
        lines.append(turn["text"])
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=False, threaded=True)
