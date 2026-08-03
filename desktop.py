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
    """Run Flask first, then defer heavy preload and market monitoring."""
    from web.app import app, start_signal_monitor

    def _preload():
        try:
            import langchain_openai  # noqa: F401
            from agents.coordinator import create_investment_agent, chat  # noqa: F401
            import chromadb  # noqa: F401
        except Exception:
            pass

    def _start_later(delay, target):
        timer = threading.Timer(delay, target)
        timer.daemon = True
        timer.start()

    # 避免 Flask 监听前的重型导入和行情监控争抢 CPU/GIL/行情锁。
    _start_later(20, _preload)
    _start_later(15, start_signal_monitor)
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
        """等待 Flask 就绪；探测失败时保留状态页，绝不跳到白屏。"""
        import urllib.request
        ready = False
        last_error = "服务尚未启动"
        for _ in range(60):  # 最多约 30 秒，每次探测都有 1 秒硬超时
            try:
                with urllib.request.urlopen("http://127.0.0.1:5000/", timeout=1) as response:
                    if response.status == 200:
                        ready = True
                        break
            except Exception as error:
                last_error = str(error)
                time.sleep(0.5)
        if ready:
            window.load_url("http://127.0.0.1:5000")
        else:
            message = repr(f"服务启动失败：{last_error}")
            window.evaluate_js(
                "document.querySelector('p').textContent=" + message + ";"
                "document.querySelector('p').style.color='#d93025';"
                "document.body.insertAdjacentHTML('beforeend',"
                "'<button onclick=\"location.href=\\\'http://127.0.0.1:5000\\\'\" "
                "style=\"padding:8px 16px;border:0;border-radius:5px;cursor:pointer\">重试</button>');"
            )

    webview.start(wait_and_load)
