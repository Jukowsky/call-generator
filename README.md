# Call Generator

AI-powered call center conversation generator with stereo TTS audio and Knovvu Analytics push integration.

## Features

- Generate realistic call center conversations using GPT-4o
- Supports English and Turkish
- Domains: Banking, E-Commerce, Telecom, Insurance, Healthcare, Utilities
- Categories: Complaint, Billing, Technical, Churn, Track Order
- Stereo TTS audio via Edge TTS (agent = left channel, customer = right channel)
- Push conversations directly to Knovvu Analytics API

## Requirements

- Python 3.13+
- FFmpeg (required by pydub for audio processing)
- OpenAI API key

## Installation

```bash
pip install -r requirements.txt
```

Install FFmpeg (Windows):
```bash
winget install --id Gyan.FFmpeg -e
```

## Running

```bash
python app.py
```

Open [http://localhost](http://localhost) in your browser.

## Settings

| Field | Description |
|-------|-------------|
| OpenAI API Key | Your OpenAI API key (`sk-...`) |
| Knovvu Identity URL | Base URL of the identity server (e.g. `https://your-tenant.identity.ca.demo.sestek.com`) |
| Knovvu Base URL | Base URL of the Analytics API (e.g. `https://your-tenant.web.ca.demo.sestek.com`) |
| Source Name | API source name configured in Knovvu Analytics (e.g. `Default Api Source`) |
| Client ID | OAuth2 client ID (e.g. `CA.Sestek.Customer`) |
| Client Secret | OAuth2 client secret |
| Agent IDs | Optional — paste specific agent UUIDs (one per line) to target. Leave blank to auto-select from API. |

## Push Flow

1. Obtains a Bearer token via OAuth2 client credentials
2. Fetches active agents from Knovvu (or uses pinned Agent IDs)
3. Synthesizes stereo audio if not already generated
4. POSTs a new call record to Knovvu Analytics
5. PUTs the audio file to the call record
