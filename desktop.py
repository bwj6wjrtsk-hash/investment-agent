"""
DataBoard Desktop App - PyWebView wrapper
Double-click to open as a standalone desktop window.
Close window = exit program cleanly.
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

# Fix proxy
for key in list(os.environ.keys()):
    if 'proxy' in key.lower():
        del os.environ[key]

from dotenv import load_dotenv
load_dotenv()


def start_flask():
    """Run Flask in background thread"""
    from web.app import app, start_signal_monitor
    start_signal_monitor()

    # Preload heavy modules in background after Flask is serving
    # This runs while user sees the UI, so first API call is instant
    def _preload():
        try:
            # langchain_openai is the heaviest single import (~2.3s)
            import langchain_openai  # noqa: F401
            # Then coordinator brings in langgraph + all tools
            from agents.coordinator import create_investment_agent, chat  # noqa: F401
            # chromadb for knowledge base
            import chromadb  # noqa: F401
        except Exception:
            pass

    threading.Thread(target=_preload, daemon=True).start()

    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    import webview

    # Start Flask server in background
    server_thread = threading.Thread(target=start_flask, daemon=True)
    server_thread.start()

    # Open window immediately with loading page, then redirect when ready
    window = webview.create_window(
        "DataBoard",
        html="""
        <html><body style="display:flex;align-items:center;justify-content:center;height:100vh;margin:0;
        font-family:-apple-system,sans-serif;background:#f8f9fa;color:#5f6368">
        <div style="text-align:center">
            <h2 style="font-weight:600;color:#202124">DataBoard</h2>
            <p>Loading...</p>
        </div>
        </body></html>
        """,
        width=1400,
        height=900,
        min_size=(1000, 600),
        text_select=True,
    )

    def wait_and_load():
        """Wait for Flask to be ready, then navigate"""
        import urllib.request
        for _ in range(30):  # max 15 seconds
            try:
                urllib.request.urlopen("http://127.0.0.1:5000/")
                break
            except:
                time.sleep(0.5)
        window.load_url("http://127.0.0.1:5000")

    webview.start(wait_and_load)
