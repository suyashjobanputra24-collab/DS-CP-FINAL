"""
PromiseGuard Website Launcher
Runs the complete FastAPI & Jinja2 interactive Web platform on http://127.0.0.1:8000
MD3135 Data Science Capstone Project
Authors: Suyash Jobanputra & Shreyash Jobanputra (Project ID: MDM-DS-039)
Guide: Dr. Kapil N. Vhatkar
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
    print("=" * 75)
    print("  🛡️  PROMISEGUARD: INTERACTIVE DATA SCIENCE WEB PLATFORM")
    print("  Third Year Computer Engineering | MD3135 Data Science (AY 2026-27)")
    print("  Project ID: MDM-DS-039 | Guide: Dr. Kapil N. Vhatkar")
    print("  Authors: Suyash Jobanputra & Shreyash Jobanputra")
    print("=" * 75)
    print("\n  Starting PromiseGuard Web Server...")
    print("  🌐 Website URL : http://127.0.0.1:8000")
    print("  ⚡ Interactive REST API Docs : http://127.0.0.1:8000/docs")
    print("\n  Opening website in your browser automatically...\n")
    print("  Press Ctrl+C in terminal to stop the server.\n")

    # Open browser automatically in background thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Launch uvicorn
    uvicorn.run("website.server:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
