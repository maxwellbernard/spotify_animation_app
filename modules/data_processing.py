"""
This module provides functions to fetch and process JSON files,
extract JSON from ZIP files, preprocess DataFrames, and prepare data for visualizations.
"""

import json
import zipfile
from datetime import datetime

import pandas as pd
import polars as pl


def fetch_and_process_files(upload_file: list) -> pd.DataFrame:
    """
    Process JSON content from either file objects or raw content (from ZIP extraction).

    Args:
        upload_files (list): List of uploaded JSON file objects OR raw JSON content from ZIP

    Returns:
        pd.DataFrame: A preprocessed DataFrame containing the necessary data
    """
    if not upload_file:
        raise FileNotFoundError(
            "No files uploaded. Please upload JSON files to continue."
        )

    needed_columns = [
        "ts",
        "ms_played",
        "master_metadata_track_name",
        "master_metadata_album_artist_name",
        "master_metadata_album_album_name",
        "spotify_track_uri",
    ]

    dfs = []
    for content in upload_file:
        try:
            if isinstance(content, bytes):
                json_data = json.loads(content.decode("utf-8"))
            elif hasattr(content, "read"):
                content.seek(0)
                json_data = json.load(content)
            elif isinstance(content, str):
                json_data = json.loads(content)
            else:
                json_data = content

            if json_data:
                df = pl.DataFrame(json_data, schema=needed_columns, strict=False)
                dfs.append(df)

        except Exception as e:
            print(f"Warning: Could not process file. Error: {e}")
            continue

    if not dfs:
        raise ValueError("No valid JSON files could be processed.")

    combined_df = pl.concat(dfs)
    processed_df = combined_df.select(
        [
            pl.col("ts")
            .str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%SZ", strict=False)
            .alias("timestamp"),
            pl.col("ms_played").alias("duration_ms"),
            pl.col("master_metadata_track_name").alias("track_name"),
            pl.col("master_metadata_album_artist_name").alias("artist_name"),
            pl.col("master_metadata_album_album_name").alias("album_name"),
            pl.col("spotify_track_uri").alias("track_uri"),
        ]
    )

    processed_df = (
        processed_df.drop_nulls()
        .filter(pl.col("duration_ms") > 30000)
        .with_columns(
            [
                pl.col("timestamp").alias("Date"),
                (pl.col("duration_ms") / 60000).alias("duration_ms"),
            ]
        )
        .drop("timestamp")
    )

    return processed_df.to_pandas()


def extract_json_from_zip(zip_file) -> list:
    """Extract JSON files from uploaded ZIP file, handling nested folder structures."""
    json_contents = []

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        file_list = zip_ref.namelist()
        json_file_names = []
        for filename in file_list:
            if (
                filename.endswith(".json")
                and "Audio" in filename  # dont want podcast or video files
                and not filename.endswith("/")
            ):
                json_file_names.append(filename)

        for json_file_name in json_file_names:
            try:
                with zip_ref.open(json_file_name) as json_file:
                    json_content = json_file.read()
                    json_contents.append(json_content)
            except Exception as e:
                print(f"Warning: Could not extract {json_file_name}: {e}")
                continue
    return json_contents


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input DataFrame by selecting relevant columns, renaming them,
    dropping rows with missing values, filtering based on duration, and converting timestamps.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    cols_to_keep = {
        "ts": "timestamp",
        "ms_played": "duration_ms",
        "master_metadata_track_name": "track_name",
        "master_metadata_album_artist_name": "artist_name",
        "master_metadata_album_album_name": "album_name",
        "spotify_track_uri": "track_uri",
    }

    df = df[list(cols_to_keep.keys())].rename(columns=cols_to_keep)
    df = df.dropna(subset=cols_to_keep.values())
    df = df[
        df["duration_ms"] > 60000
    ]  # filters out songs only played for less than 1 minute
    df["Date"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")
    df = df.drop(columns=["timestamp"])
    df["duration_ms"] = df["duration_ms"] / 60000

    return df


def prepare_df_for_visual_anims(
    df: pd.DataFrame,
    selected_attribute: str,
    analysis_metric: str,
    start_date: datetime,
    end_date: datetime,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    - Prepare the input DataFrame for animation by filtering based on the selected attribute,
    analysis metric, date range, and top N values.
    - Fill in missing dates with 0 values and calculate cumulative streams or time listened.
    Args:
        df (pd.DataFrame): Input DataFrame
        selected_attribute (str): The attribute to analyze
        (e.g., 'artist_name', 'track_name', 'album_name')
        analysis_metric (str): The metric to analyze
        (e.g., 'Number of Streams', 'Time Listened')

    Returns:
        pd.DataFrame: DataFrame with missing dates filled in
    """
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
    filter_number = 200
    if analysis_metric == "Streams":
        # each row is one stream
        if selected_attribute == "artist_name":
            top_values = (
                df.groupby("artist_name")
                .size()
                .nlargest(filter_number)
                .reset_index(name="Streams")
            )
        elif selected_attribute == "track_name":
            top_values = (
                df.groupby(["track_name", "artist_name"])
                .size()
                .nlargest(filter_number)
                .reset_index(name="Streams")
            )
        else:
            top_values = (
                df.groupby(["album_name", "artist_name"])
                .size()
                .nlargest(filter_number)
                .reset_index(name="Streams")
            )

    elif analysis_metric == "duration_ms":
        if selected_attribute == "artist_name":
            top_values = (
                df.groupby("artist_name")["duration_ms"]
                .sum()
                .nlargest(filter_number)
                .reset_index()
            )
        elif selected_attribute == "track_name":
            top_values = (
                df.groupby(["track_name", "artist_name"])["duration_ms"]
                .sum()
                .nlargest(filter_number)
                .reset_index()
            )
        else:
            top_values = (
                df.groupby(["album_name", "artist_name"])["duration_ms"]
                .sum()
                .nlargest(filter_number)
                .reset_index()
            )

    if selected_attribute == "album_name":
        top_values_list = top_values["album_name"].tolist()
    elif selected_attribute == "artist_name":
        top_values_list = top_values["artist_name"].tolist()
    else:
        top_values_list = top_values["track_name"].tolist()

    df = df[df[selected_attribute].isin(top_values_list)]

    full_date_range = pd.date_range(start=start_date, end=end_date)
    if selected_attribute == "album_name":
        unique_values = df[["album_name", "artist_name"]].drop_duplicates()
    elif selected_attribute == "artist_name":
        unique_values = df["artist_name"].unique()
    else:
        unique_values = df[["track_name", "artist_name"]].drop_duplicates()

    new_df_list = []
    if selected_attribute == "album_name":
        for _, row in unique_values.iterrows():
            album = row["album_name"]
            artist = row["artist_name"]
            subset = df[(df["album_name"] == album) & (df["artist_name"] == artist)]
            if analysis_metric == "duration_ms":
                df_grouped = (
                    subset.groupby(["album_name", "artist_name", "Date"])[
                        analysis_metric
                    ]
                    .sum()
                    .reset_index(name=analysis_metric)
                )
            else:
                df_grouped = (
                    subset.groupby(["album_name", "artist_name", "Date"])
                    .size()
                    .reset_index(name=analysis_metric)
                )

            df_grouped = (
                df_grouped.set_index("Date")
                .reindex(full_date_range, fill_value=0)  # to fill in missing dates
                .reset_index(names="Date")
            )

            df_grouped[selected_attribute] = album
            df_grouped["artist_name"] = artist
            new_df_list.append(subset)
    elif selected_attribute == "track_name":
        for _, row in unique_values.iterrows():
            track = row["track_name"]
            artist = row["artist_name"]

            subset = df[(df["track_name"] == track) & (df["artist_name"] == artist)]

            if analysis_metric == "duration_ms":
                df_grouped = (
                    subset.groupby(["track_name", "artist_name", "Date"])[
                        analysis_metric
                    ]
                    .sum()
                    .reset_index(name=analysis_metric)
                )
            else:
                df_grouped = (
                    subset.groupby(["track_name", "artist_name", "Date"])
                    .size()
                    .reset_index(name=analysis_metric)
                )

            df_grouped = (
                df_grouped.set_index("Date")
                .reindex(full_date_range, fill_value=0)
                .reset_index(names="Date")
            )

            if selected_attribute == "track_name":
                df_grouped[selected_attribute] = track
                df_grouped["artist_name"] = artist
            elif selected_attribute == "artist_name":
                df_grouped[selected_attribute] = artist
            else:
                df_grouped[selected_attribute] = album
                df_grouped["artist_name"] = artist

            new_df_list.append(subset)

    else:
        selected_attribute = "artist_name"
        for artist in unique_values:
            subset = df[df[selected_attribute] == artist]

            if analysis_metric == "duration_ms":
                df_grouped = (
                    subset.groupby(["artist_name", "Date"])[analysis_metric]
                    .sum()
                    .reset_index(name=analysis_metric)
                )
            else:
                df_grouped = (
                    subset.groupby(["artist_name", "Date"])
                    .size()
                    .reset_index(name=analysis_metric)
                )

            df_grouped = (
                df_grouped.set_index("Date")
                .reindex(full_date_range, fill_value=0)
                .reset_index(names="Date")
            )

            df_grouped[selected_attribute] = artist
            new_df_list.append(subset)

    result_df = pd.concat(new_df_list).reset_index(drop=True)
    result_df = result_df.sort_values([selected_attribute, "Date"])
    return result_df


def prepare_df_for_visual_plots(
    df: pd.DataFrame,
    selected_attribute: str,
    analysis_metric: str,
    start_date: datetime,
    end_date: datetime,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    - Prepare the input DataFrame for animation by filtering based on the selected attribute,
    analysis metric, date range, and top N values.
    - Fill in missing dates with 0 values and calculate cumulative streams or time listened.
    Args:
        df (pd.DataFrame): Input DataFrame
        selected_attribute (str): The attribute to analyze
        (e.g., 'artist_name', 'track_name', 'album_name')
        analysis_metric (str): The metric to analyze
        (e.g., 'Number of Streams', 'Time Listened')

    Returns:
        pd.DataFrame: DataFrame with missing dates filled in
    """
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
    filter_number = 10
    if analysis_metric == "Streams":
        # each row is one stream
        if selected_attribute == "artist_name":
            top_values = (
                df.groupby("artist_name")
                .size()
                .nlargest(filter_number)
                .reset_index(name="Streams")
            )
        elif selected_attribute == "track_name":
            top_values = (
                df.groupby(["track_name", "artist_name"])
                .size()
                .nlargest(filter_number)
                .reset_index(name="Streams")
            )
        else:
            top_values = (
                df.groupby(["album_name", "artist_name"])
                .size()
                .nlargest(filter_number)
                .reset_index(name="Streams")
            )

    elif analysis_metric == "duration_ms":
        if selected_attribute == "artist_name":
            top_values = (
                df.groupby("artist_name")["duration_ms"]
                .sum()
                .nlargest(filter_number)
                .reset_index()
            )
        elif selected_attribute == "track_name":
            top_values = (
                df.groupby(["track_name", "artist_name"])["duration_ms"]
                .sum()
                .nlargest(filter_number)
                .reset_index()
            )
        else:
            top_values = (
                df.groupby(["album_name", "artist_name"])["duration_ms"]
                .sum()
                .nlargest(filter_number)
                .reset_index()
            )

    top_values_list = top_values[selected_attribute].tolist()
    df = df[df[selected_attribute].isin(top_values_list)]
    full_date_range = pd.date_range(start=start_date, end=end_date)

    if selected_attribute == "album_name":
        unique_values = df[["album_name", "artist_name"]].drop_duplicates()
    elif selected_attribute == "artist_name":
        unique_values = df["artist_name"].unique()
    else:
        unique_values = df[["track_name", "artist_name"]].drop_duplicates()

    new_df_list = []
    if selected_attribute == "album_name":
        for _, row in unique_values.iterrows():
            album = row["album_name"]
            artist = row["artist_name"]
            subset = df[(df["album_name"] == album) & (df["artist_name"] == artist)]
            if analysis_metric == "duration_ms":
                df_grouped = (
                    subset.groupby(["album_name", "artist_name", "Date"])[
                        analysis_metric
                    ]
                    .sum()
                    .reset_index(name=analysis_metric)
                )
            else:
                df_grouped = (
                    subset.groupby(["album_name", "artist_name", "Date"])
                    .size()
                    .reset_index(name=analysis_metric)
                )

            df_grouped = (
                df_grouped.set_index("Date")
                .reindex(full_date_range, fill_value=0)  # to fill in missing dates
                .reset_index(names="Date")
            )

            df_grouped[selected_attribute] = album
            df_grouped["artist_name"] = artist
            new_df_list.append(subset)
    elif selected_attribute == "track_name":
        for _, row in unique_values.iterrows():
            track = row["track_name"]
            artist = row["artist_name"]

            subset = df[(df["track_name"] == track) & (df["artist_name"] == artist)]

            if analysis_metric == "duration_ms":
                df_grouped = (
                    subset.groupby(["track_name", "artist_name", "Date"])[
                        analysis_metric
                    ]
                    .sum()
                    .reset_index(name=analysis_metric)
                )
            else:
                df_grouped = (
                    subset.groupby(["track_name", "artist_name", "Date"])
                    .size()
                    .reset_index(name=analysis_metric)
                )

            df_grouped = (
                df_grouped.set_index("Date")
                .reindex(full_date_range, fill_value=0)
                .reset_index(names="Date")
            )

            if selected_attribute == "track_name":
                df_grouped[selected_attribute] = track
                df_grouped["artist_name"] = artist
            elif selected_attribute == "artist_name":
                df_grouped[selected_attribute] = artist
            else:
                df_grouped[selected_attribute] = album
                df_grouped["artist_name"] = artist

            new_df_list.append(subset)

    else:
        selected_attribute = "artist_name"
        for artist in unique_values:
            subset = df[df[selected_attribute] == artist]

            if analysis_metric == "duration_ms":
                df_grouped = (
                    subset.groupby(["artist_name", "Date"])[analysis_metric]
                    .sum()
                    .reset_index(name=analysis_metric)
                )
            else:
                df_grouped = (
                    subset.groupby(["artist_name", "Date"])
                    .size()
                    .reset_index(name=analysis_metric)
                )

            df_grouped = (
                df_grouped.set_index("Date")
                .reindex(full_date_range, fill_value=0)
                .reset_index(names="Date")
            )

            df_grouped[selected_attribute] = artist
            new_df_list.append(subset)

    result_df = pd.concat(new_df_list).reset_index(drop=True)
    result_df = result_df.sort_values([selected_attribute, "Date"])
    return result_df
