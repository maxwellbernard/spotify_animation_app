"""
This module provides functions to create a static bar plot for Spotify data analysis.
It includes functions to plot the final frame of a bar chart animation, process data,
and handle image fetching and caching.
"""

import os
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

from modules.prepare_visuals import (
    error_logged,
    fetch_images_batch,
    get_dominant_color,
    get_fonts,
    image_cache,
    setup_bar_plot_style,
)


def plot_final_frame(
    df,
    top_n,
    analysis_metric,
    selected_attribute,
    start_date,
    end_date,
    period,
    days,
    image_cache=image_cache,
    error_logged=error_logged,
) -> None:
    """Create a static plot of the final frame of the bar chart animation."""
    # Initialize image_cache and error_logged
    if image_cache is None:
        image_cache = {}
    if error_logged is None:
        error_logged = set()

    # fig setup
    fig, ax = plt.subplots(figsize=(16, 21.2), dpi=300)
    fig.patch.set_facecolor("#F0F0F0")  # Light gray
    plt.subplots_adjust(left=0.31, right=0.88, top=0.73, bottom=0.085)
    font_prop_heading, font_path_labels = get_fonts()
    title_map = {
        ("artist_name", "Streams"): "Most Played Artists",
        ("track_name", "Streams"): "Most Played Songs",
        ("album_name", "Streams"): "Most Played Albums",
        ("artist_name", "duration_ms"): "Most Played Artists",
        ("track_name", "duration_ms"): "Most Played Songs",
        ("album_name", "duration_ms"): "Most Played Albums",
    }
    title = title_map.get((selected_attribute, analysis_metric))
    fig.suptitle(
        title.format(top_n=top_n),
        y=0.93,
        x=0.53,
        fontsize=56,
        fontproperties=font_prop_heading,
    )

    # Load logo
    img_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "2024 Spotify Brand Assets",
        "Spotify_Full_Logo_RGB_Green.png",
    )
    img = mpimg.imread(img_path)
    image_axes = fig.add_axes(
        [0.38, 0.555, 0.29, 0.59]
    )  # [left, bottom, width, height]
    image_axes.imshow(img)
    image_axes.axis("off")

    # process data
    item_type = {"artist_name": "artist", "track_name": "track", "album_name": "album"}[
        selected_attribute
    ]

    df["Date"] = df["Date"].dt.to_period(period)
    if analysis_metric == "duration_ms":
        if selected_attribute == "track_name":
            monthly_df = (
                df.groupby(["Date", selected_attribute, "artist_name", "track_uri"])[
                    analysis_metric
                ]
                .sum()
                .reset_index()
            )
        elif selected_attribute == "album_name":
            monthly_df = (
                df.groupby(["Date", selected_attribute, "artist_name"])[analysis_metric]
                .sum()
                .reset_index()
            )
            track_uri_mapping = df.groupby(selected_attribute)["track_uri"].first()
            monthly_df["track_uri"] = monthly_df[selected_attribute].map(
                track_uri_mapping
            )
        else:  # artist_name
            monthly_df = (
                df.groupby(["Date", selected_attribute])[analysis_metric]
                .sum()
                .reset_index()
            )
            track_uri_mapping = df.groupby(selected_attribute)["track_uri"].first()
            monthly_df["track_uri"] = monthly_df.apply(
                lambda row: track_uri_mapping.get(
                    (row[selected_attribute], row["artist_name"])
                ),
                axis=1,
            )
        monthly_df = monthly_df.sort_values("Date")
        monthly_df[f"Cumulative_{analysis_metric}"] = monthly_df.groupby(
            selected_attribute
        )[analysis_metric].cumsum()
    elif analysis_metric == "Streams":
        if selected_attribute == "track_name":
            monthly_df = (
                df.groupby(["Date", selected_attribute, "artist_name", "track_uri"])
                .size()
                .reset_index(name="Streams")
            )
        elif selected_attribute == "album_name":
            monthly_df = (
                df.groupby(["Date", selected_attribute, "artist_name"])
                .size()
                .reset_index(name="Streams")
            )
            track_uri_mapping = df.groupby(selected_attribute)["track_uri"].first()
            monthly_df["track_uri"] = monthly_df[selected_attribute].map(
                track_uri_mapping
            )
        else:  # artist_name
            monthly_df = (
                df.groupby(["Date", selected_attribute])
                .size()
                .reset_index(name="Streams")
            )
            track_uri_mapping = df.groupby(selected_attribute)["track_uri"].first()
            monthly_df["track_uri"] = monthly_df.apply(
                lambda row: track_uri_mapping.get(
                    (row[selected_attribute], row["artist_name"])
                ),
                axis=1,
            )
        monthly_df = monthly_df.sort_values("Date")

        if analysis_metric == "Streams":
            if selected_attribute == "track_name":
                monthly_df[f"Cumulative_{analysis_metric}"] = monthly_df.groupby(
                    [selected_attribute, "artist_name", "track_uri"]
                )[analysis_metric].cumsum()
            elif selected_attribute == "album_name":
                monthly_df[f"Cumulative_{analysis_metric}"] = monthly_df.groupby(
                    [selected_attribute, "artist_name"]
                )[analysis_metric].cumsum()
            else:  # artist_name
                monthly_df[f"Cumulative_{analysis_metric}"] = monthly_df.groupby(
                    selected_attribute
                )[analysis_metric].cumsum()
        else:  # duration_ms
            monthly_df[f"Cumulative_{analysis_metric}"] = monthly_df.groupby(
                selected_attribute
            )[analysis_metric].cumsum()

    monthly_df["Date"] = monthly_df["Date"].dt.to_timestamp()
    timestamps = sorted(monthly_df["Date"].unique())
    timestamps = [ts for ts in timestamps if start_date <= ts <= end_date]
    timestamps = timestamps[::days]

    if timestamps[-1] != end_date:
        if end_date > timestamps[-1]:
            timestamps.append(end_date)
        else:
            timestamps[-1] = end_date
        timestamps.sort()

    current_time = timestamps[-1]
    cumulative_df = monthly_df[monthly_df["Date"] <= current_time]

    if selected_attribute == "track_name":
        current_df = cumulative_df.groupby(selected_attribute, as_index=False).agg(
            {
                f"Cumulative_{analysis_metric}": "max",
                "artist_name": "first",
                "track_uri": "first",
            }
        )
    elif selected_attribute == "album_name":
        current_df = cumulative_df.groupby(selected_attribute, as_index=False).agg(
            {
                f"Cumulative_{analysis_metric}": "max",
                "artist_name": "first",
                "track_uri": "first",
            }
        )
    else:  # artist_name
        current_df = cumulative_df.groupby(selected_attribute, as_index=False).agg(
            {
                f"Cumulative_{analysis_metric}": "max",
                "track_uri": "first",
            }
        )
        current_df["artist_name"] = current_df[selected_attribute]

    prev_names = current_df[selected_attribute].tolist()
    current_df["prev_rank"] = (
        current_df[selected_attribute]
        .map({name: i for i, name in enumerate(prev_names)})
        .fillna(top_n)
    )
    top_n_df = (
        current_df.sort_values(
            [f"Cumulative_{analysis_metric}", "prev_rank"],
            ascending=[False, True],
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    if top_n_df.empty:
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        plt.show()
        return

    # final frame data (no interpolation)
    widths = [0] * top_n
    labels = [""] * top_n
    names = [""] * top_n
    positions = list(range(top_n - 1, -1, -1))  # descending order

    for i, row in top_n_df.iterrows():
        widths[i] = row[f"Cumulative_{analysis_metric}"]

        if selected_attribute == "track_name" or selected_attribute == "album_name":
            song_name = "\n".join(textwrap.wrap(row[selected_attribute], width=22))
            labels[i] = song_name
        else:
            labels[i] = "\n".join(textwrap.wrap(row[selected_attribute], width=20))

        names[i] = row[selected_attribute]

    max_width = max(widths) if widths else 1

    # Calculate minimum bar width based on top_n
    min_bar_width_mapping = {
        1: 0.30,
        2: 0.54,
        3: 0.37,
        4: 0.28,
        5: 0.22,
        6: 0.19,
        7: 0.16,
        8: 0.14,
        9: 0.13,
        10: 0.11,
    }

    min_bar_width_multiplier = min_bar_width_mapping.get(top_n)
    min_bar_width = max_width * min_bar_width_multiplier
    display_widths = [max(width, min_bar_width) for width in widths]

    # plot bars
    bars = ax.barh(
        positions,
        display_widths,
        alpha=0.7,
        height=0.72,
        edgecolor="#D3D3D3",
        linewidth=1.2,
    )

    # axis styling
    ax.set_yticks([])
    ax.xaxis.label.set_fontproperties(font_path_labels)
    ax.xaxis.label.set_size(18)
    ax.xaxis.set_label_coords(-0.95, -0.05)
    setup_bar_plot_style(ax, top_n, analysis_metric)

    # year and month text
    ax.text(
        0.38,
        1.10,
        f"{start_date.year} {start_date.strftime('%B')} - {current_time.year} {current_time.strftime('%B')}",
        transform=ax.transAxes,
        fontsize=36,
        fontproperties=font_prop_heading,
        bbox=dict(facecolor="#F0F0F0", edgecolor="none", alpha=0.7),
        color="#A9A9A9",
        ha="center",
        va="top",
    )
    ax.text(
        0.38,
        -0.033,
        "Streams" if analysis_metric == "Streams" else "Minutes Listened",
        transform=ax.transAxes,
        fontsize=28,
        fontproperties=font_prop_heading,
        bbox=dict(facecolor="#F0F0F0", edgecolor="none", alpha=0.7),
        color="#A9A9A9",
        ha="center",
        va="top",
    )

    # Image scaling and positioning
    top_n_scale_mapping_height = {
        1: 760,
        2: 390,
        3: 280,
        4: 210,
        5: 165,
        6: 137,
        7: 115,
        8: 101,
        9: 93,
        10: 84,
    }

    top_n_xybox_mapping = {
        1: (-295, 0),
        2: (-160, 0),
        3: (-112, 0),
        4: (-85, 0),
        5: (-67, 0),
        6: (-57, 0),
        7: (-48, 0),
        8: (-43, 0),
        9: (-39, 0),
        10: (-35, 0),
    }
    bar_height = 0.7
    scale_factor = top_n_scale_mapping_height.get(top_n)
    target_size = int(bar_height * scale_factor)

    def process_images_with_batch_api(
        names, monthly_df, selected_attribute, item_type, top_n
    ) -> None:
        """
        Process images using batch API + parallel downloads for efficiency.
        """
        items_to_fetch = []
        cache_keys = []

        for name in names:
            cache_key = f"{name}_top_n_{top_n}"
            cache_keys.append(cache_key)

            if cache_key not in image_cache:
                row = monthly_df[monthly_df[selected_attribute] == name].iloc[0]
                item_data = {"name": name, "type": item_type, "cache_key": cache_key}

                if "track_uri" in row and row["track_uri"]:
                    item_data["track_uri"] = row["track_uri"]
                else:
                    image_cache[cache_key] = None
                    if item_type == "artist":
                        item_data["artist_name"] = name
                        item_data["search_required"] = True

                items_to_fetch.append(item_data)

        # batch API calls
        if items_to_fetch:
            batch_items = [
                item for item in items_to_fetch if not item.get("search_required")
            ]
            search_items = [
                item for item in items_to_fetch if item.get("search_required")
            ]

            batch_results = {}

            if batch_items:
                batch_results = fetch_images_batch(batch_items)
            if search_items:
                from modules.prepare_visuals import fetch_image

                for item in search_items:
                    try:
                        image_url = fetch_image(item["name"], "artist")
                        if image_url:
                            batch_results[item["name"]] = image_url
                        time.sleep(0.1)
                    except Exception as e:
                        print(f"Search failed for {item['name']}: {e}")

            # prepare download tasks
            download_tasks = []
            for item in items_to_fetch:
                image_url = None

                if item["type"] == "track":
                    image_url = batch_results.get(
                        item.get("track_uri")
                    ) or batch_results.get(item["name"])
                elif item["type"] == "album":
                    image_url = batch_results.get(item["name"])
                elif item["type"] == "artist":
                    image_url = batch_results.get(item["name"])

                if image_url:
                    download_tasks.append(
                        {
                            "name": item["name"],
                            "cache_key": item["cache_key"],
                            "image_url": image_url,
                        }
                    )
                else:
                    image_cache[item["cache_key"]] = None

            # download images in parallel
            if download_tasks:
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [
                        executor.submit(_download_and_cache_image, task)
                        for task in download_tasks
                    ]

                    successful_downloads = 0
                    for future in futures:
                        if future.result():
                            successful_downloads += 1

        # Handle already cached items
        for name, cache_key in zip(names, cache_keys):
            if cache_key in image_cache:
                pass

    def _download_and_cache_image(task) -> bool:
        """Download and cache a single image - designed for parallel execution"""
        name = task["name"]
        cache_key = task["cache_key"]
        image_url = task["image_url"]

        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            img_resized = img.resize(
                (target_size, target_size), Image.Resampling.LANCZOS
            )
            color = get_dominant_color(img_resized, name)
            image_cache[cache_key] = {"img": img_resized, "color": color}
            return True
        except Exception:
            image_cache[cache_key] = None
            return False

    process_images_with_batch_api(
        names, monthly_df, selected_attribute, item_type, top_n
    )

    # Create text, label, and image annotation objects
    text_objects = [None] * top_n
    label_objects = [None] * top_n
    image_annotations = [None] * top_n

    # font size mapping for labels based on top_n
    if selected_attribute in ["track_name", "album_name"]:
        top_n_label_fontsize_mapping = {
            1: 22,
            2: 22,
            3: 22,
            4: 22,
            5: 22,
            6: 20,
            7: 20,
            8: 20,
            9: 19,
            10: 19,
        }
        label_fontsize = top_n_label_fontsize_mapping.get(top_n, 22)
    else:
        label_fontsize = 22  # for artist_name, use fixed font size

    for i in range(top_n):
        name = names[i]
        text_x = display_widths[i]
        text_y = positions[i]
        max_value = max(display_widths)
        offset = max(0.01, max_value * 0.03)

        # numbers on bar
        text_objects[i] = ax.text(
            text_x + offset,
            text_y,
            f"{widths[i]:,.0f}",
            va="center",
            ha="left",
            fontsize=24,
            fontproperties=font_path_labels,
        )

        # y-axis labels
        label_objects[i] = ax.text(
            -offset,
            text_y,
            labels[i],
            va="center",
            ha="right",
            fontsize=label_fontsize,  # Use dynamic font size
            fontproperties=font_path_labels,
        )

        if selected_attribute == "track_name" or selected_attribute == "album_name":
            current_row = top_n_df.iloc[i]
            artist_name = f"({current_row['artist_name']})"
            artist_wrapped = "\n".join(textwrap.wrap(artist_name, width=25))
            song_lines = labels[i].count("\n") + 1

            # spacing values for each top_n and number of lines
            line_spacing_mapping = {
                1: {1: 0.06, 2: 0.10, 3: 0.22},
                2: {1: 0.08, 2: 0.12, 3: 0.14},
                3: {1: 0.10, 2: 0.14, 3: 0.19},
                4: {1: 0.14, 2: 0.19, 3: 0.25},
                5: {1: 0.16, 2: 0.23, 3: 0.29},
                6: {1: 0.17, 2: 0.24, 3: 0.32},
                7: {1: 0.20, 2: 0.29, 3: 0.36},
                8: {1: 0.22, 2: 0.31, 3: 0.39},
                9: {1: 0.24, 2: 0.33, 3: 0.43},
                10: {1: 0.25, 2: 0.35, 3: 0.45},
            }
            top_n_spacing = line_spacing_mapping.get(top_n, {})
            artist_y_offset = top_n_spacing.get(song_lines, 0.30)

            # y-axis subtext
            ax.text(
                -offset,
                text_y - artist_y_offset,
                artist_wrapped,
                va="center",
                ha="right",
                fontsize=label_fontsize - 2,
                fontproperties=font_path_labels,
                color="#A9A9A9",  # grey
            )

        # add image
        cache_key = f"{name}_top_n_{top_n}"
        img_data = image_cache.get(cache_key)
        if img_data and text_x > 0:
            img = img_data["img"]
            xybox = top_n_xybox_mapping.get(top_n)
            if img_data["color"]:
                bars[i].set_facecolor(np.array(img_data["color"]) / 255)

            image_width_estimate = abs(xybox[0]) if xybox else 50
            min_x_position = image_width_estimate / max_value * 0.8
            image_x_position = max(text_x, min_x_position)
            img_box = OffsetImage(img)

            # calculate the y position for the image
            image_annotations[i] = AnnotationBbox(
                img_box,
                (image_x_position, text_y),
                xybox=xybox,
                xycoords="data",
                boxcoords="offset points",
                frameon=False,
                bboxprops=dict(
                    boxstyle="round,pad=0.05",
                    edgecolor="#A9A9A9",
                    facecolor="#DCDCDC",
                    linewidth=0.5,
                ),
            )
            ax.add_artist(image_annotations[i])
            image_annotations[i].set_visible(True)

    # axis limits
    ax.set_xlim(0, max(display_widths) * 1.1)
    ax.set_ylim(-0.6, top_n - 0.4)

    return fig
