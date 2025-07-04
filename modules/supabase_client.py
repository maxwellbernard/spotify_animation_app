"""
This module provides functions to interact with the Supabase database,
including querying, inserting, and updating data.
It uses the Supabase Python client to connect to the database.
Supabase is used as a backend for monitoring the web app user activity.
"""

import streamlit as st
from supabase import create_client

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
