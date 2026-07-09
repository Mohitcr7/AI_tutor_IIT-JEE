# Deploying the JEE AI Tutor

The app is a single Streamlit process. It needs two things at runtime:

1. `GEMINI_API_KEY` — the Google Gemini API key that powers the tutor
   (free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)).
2. The ChromaDB vector store — built automatically on first boot from the
   committed source JSON (concept graphs + PYQ corpus), so you do **not** need
   to commit the store itself.

---

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # then edit .env and paste your key
python embeddings/build_concept_store.py   # optional: prebuild the store (~1 min)

streamlit run app.py
```

If you skip the prebuild step, the app builds the store on the first question
instead (one-time, ~1–2 min while it downloads the embedding model and embeds
the 877 documents).

---

## Deploy to Streamlit Community Cloud (recommended, free)

1. Push this repo to GitHub (the vector store and `.env` are gitignored — that
   is correct; they must not be committed).
2. Go to [share.streamlit.io](https://share.streamlit.io), "New app", point it
   at your repo and `app.py`.
3. In **App settings → Secrets**, add your key in TOML form:
   ```toml
   GEMINI_API_KEY = "AIza..."
   ```
   The app reads this via `st.secrets` automatically.
4. Deploy. First boot builds the vector store (watch the logs); subsequent boots
   reuse it.

**Faster first boot (optional):** commit the prebuilt store so there is no build
on deploy. Remove `vectorstore/` from `.gitignore`, run
`python embeddings/build_concept_store.py`, and commit `vectorstore/`. It is
~7.6 MB. The trade-off: you must rebuild and recommit whenever the concept
graphs or PYQ corpus change.

---

## Deploy to Hugging Face Spaces (alternative)

Create a **Streamlit** Space, push this repo, and add `GEMINI_API_KEY` under
**Settings → Variables and secrets**. Same auto-build behavior applies.

---

## Operational must-haves before real students use it

- **Never commit the key.** It lives only in `.env` (local) or the host's
  secrets store (deployed). Rotate it if it ever leaks.
- **Mind the free-tier rate limits.** Gemini's free tier caps requests per
  minute and per day. That is fine for a beta, but a class of students hitting
  it at once can exhaust the per-minute quota — expect occasional 429s under
  load. Upgrade to a paid Gemini tier (or add request queuing) before scaling.
- **Rate-limit per user.** Streamlit has no built-in auth; students can send
  many requests. Add a simple per-session throttle (e.g. cap questions per
  minute in `st.session_state`) before opening it up widely — this also protects
  your free-tier quota.
- **Pick the model deliberately.** `TUTOR_MODEL` (env var) overrides the default
  `gemini-2.5-flash-lite` (fastest, most generous free quota). Step up to
  `gemini-2.5-flash` or `gemini-2.5-pro` for deeper reasoning — tighter free-tier
  limits.
