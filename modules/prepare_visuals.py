"""
This module provides functions to prepare visuals for Spotify data analysis,
including fetching images, extracting dominant colors, and setting up plot styles.
"""

import colorsys
import os
import time
from io import BytesIO
from typing import Dict, List

import matplotlib.pyplot as plt
import spotipy
import streamlit as st
from colorthief import ColorThief
from matplotlib.font_manager import FontProperties
from PIL import Image
from spotipy.oauth2 import SpotifyClientCredentials

# global caches and eror tracking
color_cache = {}
image_cache = {}
error_logged = set()

# load environment variables
client_id = st.secrets["SPOTIFY_CLIENT_ID"]
client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"]

client_credentials_manager = SpotifyClientCredentials(
    client_id=client_id, client_secret=client_secret
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def fetch_images_batch(items_data: List[Dict]) -> Dict[str, str]:
    """
    Fetch images in batches using Spotify's batch endpoints.
    """
    image_urls = {}

    tracks = []
    albums = []
    artists = []

    for item in items_data:
        if item["type"] == "track" and item.get("track_uri"):
            tracks.append(item)
        elif item["type"] == "album" and item.get("track_uri"):
            albums.append(item)
        elif item["type"] == "artist" and item.get("track_uri"):
            artists.append(item)

    if tracks:
        track_uris = [item["track_uri"] for item in tracks]
        track_images = _fetch_tracks_batch(track_uris)
        image_urls.update(track_images)

    if albums:
        track_uris = [item["track_uri"] for item in albums]
        album_images = _fetch_tracks_batch(track_uris)
        for item in albums:
            track_uri = item["track_uri"]
            if track_uri in album_images:
                album_name = item["name"]
                image_urls[album_name] = album_images[track_uri]

    if artists:
        artist_images = _fetch_artists_from_tracks_batch(artists)
        image_urls.update(artist_images)

    return image_urls


def _fetch_artists_from_tracks_batch(artist_items: List[Dict]) -> Dict[str, str]:
    """
    Fetch artist images using track URIs in batches.
    Step 1: Get track info (batch) then extract artist IDs
    Step 2: Get artist info (batch) then extract images
    """
    image_urls = {}
    tracks_api_calls = 0
    artists_api_calls = 0

    # we only have artist_name from metadata, so we need to get the artist_id from the
    # track_uri to be able to process the artist images in batches.

    # extract track IDs from URIs
    track_uris = [item["track_uri"] for item in artist_items]
    track_ids = []
    uri_to_name = {}

    for item in artist_items:
        track_uri = item["track_uri"]
        track_id = track_uri.split(":")[-1] if ":" in track_uri else track_uri
        track_ids.append(track_id)
        uri_to_name[track_uri] = item["name"]  # store the artist name by track URI

    # batch fetch track information
    artist_id_to_name = {}
    all_artist_ids = []

    for i in range(0, len(track_ids), 50):
        batch_track_ids = track_ids[i : i + 50]
        batch_track_uris = track_uris[i : i + 50]

        try:
            print(
                f"🚀 Fetching track batch {i // 50 + 1}: {len(batch_track_ids)} tracks"
            )
            tracks_response = sp.tracks(batch_track_ids)
            tracks_api_calls += 1

            for j, track in enumerate(tracks_response["tracks"]):
                if track and track.get("artists"):
                    track_uri = batch_track_uris[j]
                    artist_name = uri_to_name.get(track_uri)

                    for artist in track["artists"]:
                        if artist["name"] == artist_name:
                            artist_id = artist["id"]
                            artist_id_to_name[artist_id] = artist_name
                            all_artist_ids.append(artist_id)
                            break

            time.sleep(0.1)

        except Exception as e:
            print(f"Batch tracks API failed: {e}")
            continue

    # batch fetch artist information
    if all_artist_ids:
        unique_artist_ids = list(dict.fromkeys(all_artist_ids))

        for i in range(0, len(unique_artist_ids), 50):
            batch_artist_ids = unique_artist_ids[i : i + 50]

            try:
                artists_response = sp.artists(batch_artist_ids)
                artists_api_calls += 1

                for artist in artists_response["artists"]:
                    if artist and artist.get("images"):
                        artist_id = artist["id"]
                        artist_name = artist_id_to_name.get(artist_id)
                        if artist_name:
                            image_url = artist["images"][0]["url"]
                            image_urls[artist_name] = image_url

                time.sleep(0.1)

            except Exception as e:
                print(f"Batch artists API failed: {e}")
                continue
    return image_urls


def _fetch_tracks_batch(track_uris: List[str]) -> Dict[str, str]:
    """Fetch track images in batches of 50"""
    image_urls = {}

    for i in range(0, len(track_uris), 50):
        batch = track_uris[i : i + 50]
        try:
            tracks_response = sp.tracks(batch)
            for track in tracks_response["tracks"]:
                if track and track["album"].get("images"):
                    image_urls[track["uri"]] = track["album"]["images"][0]["url"]
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 429:
                retry_after = int(e.headers.get("Retry-After", 5))
                print(f"Spotify Rate Limit: Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
                # Retry this batch
                return _fetch_tracks_batch(track_uris[i:])
            print(f"Error fetching tracks batch: {e}")
        time.sleep(0.1)
    return image_urls


def _fetch_albums_batch(album_ids: List[str]) -> Dict[str, str]:
    """Fetch album images in batches of 20"""
    image_urls = {}

    for i in range(0, len(album_ids), 20):
        batch = album_ids[i : i + 20]
        try:
            albums_response = sp.albums(batch)
            for album in albums_response["albums"]:
                if album and album.get("images"):
                    image_urls[album["id"]] = album["images"][0]["url"]
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 429:
                retry_after = int(e.headers.get("Retry-After", 5))
                print(f"Spotify Rate Limit: Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
                return _fetch_albums_batch(album_ids[i:])
            print(f"Error fetching albums batch: {e}")
        time.sleep(0.1)
    return image_urls


def fetch_image(
    item_name: str, item_type: str, artist_name: str = None, track_uri: str = None
) -> str:
    """Fetches the image using track_uri for tracks/albums, or search for artists."""
    try:
        if item_type == "artist":
            result = sp.search(q=f"artist:{item_name}", type="artist", limit=1)
            if result["artists"]["items"]:
                images = result["artists"]["items"][0].get("images", [])
                return images[0]["url"] if images else None

        elif item_type in ["track"] and track_uri:
            try:
                track = sp.track(track_uri)
                return (
                    track["album"]["images"][0]["url"]
                    if track["album"].get("images")
                    else None
                )
            except spotipy.exceptions.SpotifyException as e:
                if e.http_status == 429:  # Rate limit error
                    retry_after = int(e.headers.get("Retry-After", 5))
                    print(f"Rate limit hit. Retrying after {retry_after} seconds...")
                    print(
                        f"Spotify Rate Limit: Retrying after {retry_after} seconds..."
                    )
                    time.sleep(retry_after)
                    return fetch_image(item_name, item_type, artist_name, track_uri)
                print(f"Spotify API error: {e}")
                return None

        elif item_type == "album":
            query = f"album:{item_name}" + (
                f" artist:{artist_name}" if artist_name else ""
            )
            result = sp.search(q=query, type="album", limit=1)
            if result["albums"]["items"]:
                images = result["albums"]["items"][0].get("images", [])
                return images[0]["url"] if images else None

        return None

    except (KeyError, IndexError) as e:
        print(f"Data error fetching image for {item_name} ({item_type}): {str(e)}")
        return None
    except Exception as e:
        print(
            f"Unexpected error fetching image for {item_name} ({item_type}): {str(e)}"
        )
        return None


def get_dominant_color(img: Image, img_name: str) -> tuple:
    """
    Extracts a vibrant dominant color from an image using ColorThief, avoiding greys.

    Args:
        img: The image to analyze.
        img_name: Unique identifier for caching.

    Returns:
        tuple: RGB color (r, g, b) between 0-255.
    """
    if img_name in color_cache:
        return color_cache[img_name]

    with BytesIO() as byte_stream:
        img.save(byte_stream, format="PNG")
        byte_stream.seek(0)
        color_thief = ColorThief(byte_stream)
        palette = color_thief.get_palette(color_count=5, quality=5)

    vibrant_colors = []
    for rgb in palette:
        # Convert RGB (0-255) to HSV (h: 0-1, s: 0-1, v: 0-1)
        h, s, v = colorsys.rgb_to_hsv(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)

        # Filter out low-saturation (grey-like) colors
        # s < 0.2 is very grey; v < 0.2 is too dark; v > 0.95 might be too white
        if s > 0.3 and 0.2 < v < 0.95:
            vibrant_colors.append(rgb)

    # If no vibrant colors found, fall back to the first palette color
    dominant_color = vibrant_colors[0] if vibrant_colors else palette[0]

    color_cache[img_name] = dominant_color
    return dominant_color


def get_fonts() -> tuple:
    """Load custom fonts for the plot.
    Returns: tuple of FontProperties for headings and labels.
    """
    font_path_heading = os.path.join(os.getcwd(), "fonts", "Montserrat-Bold.ttf")
    font_path_labels = os.path.join(os.getcwd(), "fonts", "Montserrat-SemiBold.ttf")
    font_prop_heading = FontProperties(
        family="sans-serif",
        style="normal",
        variant="normal",
        weight="normal",
        stretch="normal",
        size="medium",
        fname=font_path_heading,
    )
    font_path_labels = FontProperties(
        family="sans-serif",
        style="normal",
        variant="normal",
        weight="normal",
        stretch="normal",
        size="medium",
        fname=font_path_labels,
    )
    return font_prop_heading, font_path_labels


def setup_bar_plot_style(
    ax: plt.Axes,
    top_n: int = 10,
    analysis_metric: str = "Streams",
) -> None:
    """Apply consistent plot styling"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_linewidth(3)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_color("grey")
    ax.spines["bottom"].set_color("grey")
    ax.margins(x=0.05)
    ax.set_title(" ", pad=200, fontsize=14, fontweight="bold")
    ax.xaxis.labelpad = 30
    ax.title.set_position([0.5, 1.3])
    ax.title.set_fontsize(20)
    ax.set_xticks([])
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)
    return None


def setup_line_plot_style(
    ax: plt.Axes,
    top_n: int = 10,
    analysis_metric: str = "Streams",
) -> None:
    """Apply consistent plot styling"""
    ax.set_xlabel("Date", fontsize=18)
    ax.set_ylabel(f"Cumulative {analysis_metric}", fontsize=18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_color("grey")
    ax.spines["bottom"].set_color("grey")
    ax.margins(x=0.05)
    ax.set_title(" ", pad=200, fontsize=14, fontweight="bold")
    ax.xaxis.labelpad = 30
    ax.grid(False)
    ax.title.set_position([0.5, 1.3])
    ax.title.set_fontsize(20)
    ax.set_facecolor("#F0F0F0")  # off white
    return None
