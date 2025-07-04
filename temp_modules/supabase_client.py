"""
This module provides functions to interact with the Supabase database,
including querying, inserting, and updating data.
It uses the Supabase Python client to connect to the database.
Supabase is used as a backend for monitoring the web app user activity.
"""

import os

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
