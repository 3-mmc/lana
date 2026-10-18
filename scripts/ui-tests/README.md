# Headless UI tests

`ui-final.mjs` drives the page end to end on the demo clips against the desktop server with no
window: recognition with confidence, the To check pass, alternatives, re-timing, reload, warnings,
and a TextGrid round trip. It needs zipa-small and omnilingual-300m in the models folder.

    cd src-tauri && cargo build
    LANA_SERVE_ONLY=47710 LANA_PROVIDERS=<file holding {"providers":[]}> LANA_STORE=<empty folder> \
      LANA_MODELS=<models folder> <target>/debug/lana-desktop &
    LANA_UI_OUT=<folder> PLAYWRIGHT_DIR=<folder with node_modules/playwright-core> \
      CHROMIUM=<chrome binary> node scripts/ui-tests/ui-final.mjs

An empty providers file keeps it from reaching any recognition server.
