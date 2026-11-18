"""
PromiseGuard™ Enterprise: Web Application & Fleet Console Launcher
Runs the complete FastAPI & Jinja2 enterprise web platform on http://127.0.0.1:8000
Real-Time Stream Inference, Conformal Uncertainty & Autonomous SLA Defense Engine
Ready for Fleet Deployment: Zomato • Swiggy • Zepto • Blinkit • DoorDash
"""

import os
import sys
import threading
import time
import webbrowser
import uvicorn

def open_browser():
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:8000")

def main():
    print("=" * 78)
    print("  🛡️  PROMISEGUARD™ ENTERPRISE: AUTONOMOUS DELIVERY SLA DEFENSE PLATFORM")
    print("  High-Volume Logistics Intelligence Engine (v3.2.0-enterprise)")
    print("  Engineered for: Zomato • Swiggy • Zepto • Blinkit • DoorDash")
    print("=" * 78)
    print("\n  Starting PromiseGuard Enterprise Web Server...")
    print("  🌐 Enterprise Console URL  : http://127.0.0.1:8000")
    print("  ⚡ Interactive OpenAPI Docs : http://127.0.0.1:8000/docs")
    print("\n  Opening console in your browser automatically...\n")
    print("  Press Ctrl+C in terminal to stop the server.\n")

    # Open browser automatically in background thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Launch uvicorn
    uvicorn.run("website.server:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
