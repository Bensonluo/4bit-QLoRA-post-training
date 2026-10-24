# Cloud display

The cloud site runs `ui/app.py`, exactly as the local workbench. There is no separate frontend. `TUNESMITH_DISPLAY_ONLY=1` disables mutations, uploads, model execution and external Agent calls; filters, navigation and existing results remain available. Local launches leave this variable unset for normal operation.

One CPU-only Streamlit container serves `/qlora-dashboard/`. It has no GPU, model weights, API keys or MLflow server. The MLflow client reads a local SQLite store; bundled public validation records provide historical results, not current cloud training. `seed_display.py` imports repository evidence once.

```sh
# From a clean source checkout (no local outputs, models or credentials in build context)
docker compose -f deploy/compose.yml up -d --build
```

Include `deploy/nginx-tunesmith.conf` in the existing HTTPS server after the health check passes. Update by building the next source version, checking health and reloading Nginx only if its configuration changed. Keep the previous image/config for rollback. The source commit and bundle digest are recorded in `deploy/source-version.json` for each release; this generated file is deployment metadata, not a second source tree.
