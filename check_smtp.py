"""Throwaway check: can we log in to the mail server?

    python check_smtp.py

Logs in and disconnects. Sends nothing. Delete this file once the batch UI
exists.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv()

from utils.email_sender import verify_connection

result = verify_connection()

if result["ok"]:
    print("SMTP connection OK")
    print(f"  host : {result['host']}:{result['port']}")
    print(f"  from : {result['from']}")
    print("\nNothing was sent - this only logs in and disconnects.")
else:
    print("SMTP connection FAILED")
    print(f"  {result['error']}")
    sys.exit(1)
