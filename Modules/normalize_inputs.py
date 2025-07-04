"""
Normalize the selected attribute and analysis metric.
This module provides a mapping from user-friendly names to internal column names
for attributes and metrics used in the Spotify data analysis.
"""

ATTRIBUTE_MAP = {
    "Artist": "artist_name",
    "Song": "track_name",
    "Album": "album_name",
}

METRIC_MAP = {
    "Number of Streams": "Streams",
    "Time Listened": "duration_ms",
}


def normalize_inputs(selected_attribute, analysis_metric):
    """Normalize attribute and metric to internal column names."""
    norm_attr = ATTRIBUTE_MAP.get(selected_attribute, selected_attribute)
    norm_metric = METRIC_MAP.get(analysis_metric, analysis_metric)
    return norm_attr, norm_metric
