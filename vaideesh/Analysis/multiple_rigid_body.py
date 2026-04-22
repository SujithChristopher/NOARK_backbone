"""this code is written by Sujith, extended for 3 rigid bodies"""

from datetime import datetime, timedelta

import msgpack
import numpy as np
import pandas as pd
from more_itertools import locate
from scipy.interpolate import interp1d


# ─────────────────────────────────────────────────────────────────────────────
# ORIGINAL FUNCTIONS (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def read_df_csv(filename, offset=2):
    pth = filename
    raw = pd.read_csv(pth)
    cols_list = raw.columns
    inx = [i for i, x in enumerate(cols_list) if x == "Capture Start Time"]
    st_time = cols_list[inx[0] + 1]
    st_time = datetime.strptime(st_time, "%Y-%m-%d %I.%M.%S.%f %p")

    mr_inx = pd.read_csv(pth, skiprows=3)
    markers_raw = mr_inx.columns
    marker_offset = offset
    markers_raw = markers_raw[marker_offset:]
    col_names = []
    for i in range(0, len(markers_raw), 3):
        col_names.append(markers_raw[i].split(":")[1])

    df_headers = ["frame", "seconds"]
    for id, i in enumerate(col_names):
        if not i.islower():
            col_names[id] = i.lower()
    for i in col_names:
        df_headers.append(i + "_x")
        df_headers.append(i + "_y")
        df_headers.append(i + "_z")

    mo_data = pd.read_csv(pth, skiprows=6)
    mo_data.columns = df_headers
    return mo_data, st_time


def add_datetime_col(df, _time, _name):
    _t = []
    for i in list(df[_name]):
        _t.append(_time + timedelta(0, float(i)))
    df["time"] = _t
    return df


def add_datetime_diff(df, _time, _sync, _diff_name, truncate=False):
    _inx = 0
    for inx, i in enumerate(df[_sync]):
        if i == 1:
            _inx = inx
            break
    df = df.loc[_inx:].copy()
    _diff = list(df[_diff_name].diff())
    _sum = 0
    _t = []
    for i in _diff:
        if np.isnan(i):
            i = 0
        _sum = _sum + i
        _t.append(_time + timedelta(0, float(_sum / 1000)))
    df["time"] = _t
    if truncate:
        _count = 0
        for count, j in enumerate(df[_sync]):
            if j == 0:
                _count = count
                break
        if _count != 0:
            df = df.loc[:_count].copy()
    return df


def add_time_from_file(df, _pth):
    with open(_pth, "rb") as f:
        _time_obj = msgpack.Unpacker(f)
        _time = []
        for i in _time_obj:
            _time.append(i)
    df["time"] = _time
    df["time"] = pd.to_datetime(df["time"])
    return df


def interpolate_target_df(target_df, reference_df, col_names=None):
    if type(target_df) is not pd.DataFrame:
        target_df = target_df.to_pandas()
    if type(reference_df) is not pd.DataFrame:
        reference_df = reference_df.to_pandas()

    target_df    = target_df.reset_index(drop=True)
    reference_df = reference_df.reset_index(drop=True)

    if col_names is None:
        col_names = ["frame_id", "x", "y", "z", "yaw", "pitch", "roll"]

    df = pd.DataFrame(columns=col_names)
    df["time"] = reference_df.time

    reference_df["time"] = (
        reference_df["time"].dt.hour * 3600
        + reference_df["time"].dt.minute * 60
        + reference_df["time"].dt.second
        + reference_df["time"].dt.microsecond / 1000000
    )
    target_df["time"] = (
        target_df["time"].dt.hour * 3600
        + target_df["time"].dt.minute * 60
        + target_df["time"].dt.second
        + target_df["time"].dt.microsecond / 1000000
    )

    new_cols = []
    for i in col_names:
        f = interp1d(target_df.time, target_df[i], fill_value="extrapolate")
        new_cols.append(f(reference_df.time))

    for idx, i in enumerate(col_names):
        df[i] = new_cols[idx]

    return df


def trunkate_dfs(df_1, df_2, display_print=False):
    if df_1["time"].iloc[0] < df_2["time"].iloc[0]:
        _start_inx = df_1.time.searchsorted(df_2.time[0])
        df_1 = df_1.loc[_start_inx:].reset_index(drop=True)
        if display_print:
            print(f"df_1 starts earlier, trimmed from index {_start_inx}")
    else:
        _start_inx = df_2.time.searchsorted(df_1.time[0])
        df_2 = df_2.loc[_start_inx:].reset_index(drop=True)
        if display_print:
            print(f"df_2 starts earlier, trimmed from index {_start_inx}")

    if df_1["time"].iloc[-1] > df_2["time"].iloc[-1]:
        _end_inx = df_1.time.searchsorted(df_2.time.iloc[-1])
        df_1 = df_1.loc[:_end_inx].reset_index(drop=True)
        if display_print:
            print(f"df_1 ends later, trimmed to index {_end_inx}")
    else:
        _end_inx = df_2.time.searchsorted(df_1.time.iloc[-1])
        df_2 = df_2.loc[:_end_inx].reset_index(drop=True)
        if display_print:
            print(f"df_2 ends later, trimmed to index {_end_inx}")

    return df_1, df_2


def read_rigid_body_csv(_pth):
    df = pd.read_csv(_pth, skiprows=2, header=None, dtype=str)

    raw_df    = pd.read_csv(_pth, dtype=str)
    cols_list = raw_df.columns
    inx       = [i for i, x in enumerate(cols_list) if x == "Capture Start Time"]
    st_time   = datetime.strptime(cols_list[inx[0] + 1], "%Y-%m-%d %I.%M.%S.%f %p")

    _marker_type = [df[idx][0] for idx in df.columns]

    def find_indices(lst, val):
        return list(locate(lst, lambda x: x == val))

    _rb_idx        = find_indices(_marker_type, "Rigid Body")
    _rb_marker_idx = find_indices(_marker_type, "Rigid Body Marker")
    _marker_idx    = find_indices(_marker_type, "Marker")

    _rb_idx.sort(); _rb_marker_idx.sort(); _marker_idx.sort()

    _rb_df         = df[1:]
    _analysis_type = list(_rb_df.iloc[2].values)
    _rotation_ids  = find_indices(_analysis_type, "Rotation")
    _rotation_ids.sort()

    _rb_pos_idx = _rb_idx.copy()
    [_rb_pos_idx.remove(i) for i in _rotation_ids]

    col_names = ["frame", "seconds"]

    for i in _rotation_ids:
        col_names.append("rb_ang_" + _rb_df[i].iloc[3].lower())

    for i in _rb_pos_idx:
        if isinstance(_rb_df[i].iloc[3], str):
            col_names.append("rb_pos_" + _rb_df[i].iloc[3].lower())
        else:
            col_names.append("rb_pos_err")

    for i in _rb_marker_idx:
        if isinstance(_rb_df[i].iloc[0], str):
            _col_head = _rb_df[i].iloc[0].lower().split(":")[1].strip().replace("marker", "")
            if isinstance(_rb_df[i].iloc[3], str):
                col_names.append("rb_marker_m" + _col_head + "_" + _rb_df[i].iloc[3].lower())
            else:
                col_names.append("rb_marker_m" + _col_head + "_mq")

    for i in _marker_idx:
        if isinstance(_rb_df[i].iloc[0], str):
            _col_head = _rb_df[i].iloc[0].lower().split(":")[1].strip().replace("marker", "")
            if isinstance(_rb_df[i].iloc[3], str):
                col_names.append("m" + _col_head + "_" + _rb_df[i].iloc[3].lower())

    _rb_df = _rb_df[4:]
    _rb_df.columns = col_names
    _rb_df = _rb_df.reset_index(drop=True)
    _rb_df = _rb_df.apply(pd.to_numeric, errors="ignore")

    return _rb_df, st_time


def get_marker_name(val):
    return {"x": f"m{val}_x", "y": f"m{val}_y", "z": f"m{val}_z"}


def get_rb_marker_name(val):
    return {ax: f"rb_marker_m{val}_{ax}" for ax in ("x", "y", "z")}


# ─────────────────────────────────────────────────────────────────────────────
# NEW: 3 RIGID BODY SUPPORT
# ─────────────────────────────────────────────────────────────────────────────
#
# Your CSV header structure (after skiprows=2):
#
#   raw row 0  →  Type row     : "Rigid Body" | "Rigid Body Marker" | "Marker"
#   raw row 1  →  Name row     : "tframe" | "tframe:Marker1" | "noark" | …
#   raw row 2  →  ID row       : UUID strings
#   raw row 3  →  Data-type    : "Rotation" | "Position" | "Mean Marker Error"
#                                | "Marker Quality"
#   raw row 4  →  Axis row     : "X" | "Y" | "Z" | "W"  (col headers in CSV)
#   raw row 5+ →  numeric data
#
# Columns 0-1 are always Frame and Time (Seconds).
# ─────────────────────────────────────────────────────────────────────────────

def _parse_capture_start_time(_pth):
    """Extract capture start time from the Motive CSV metadata row."""
    raw  = pd.read_csv(_pth, nrows=0, dtype=str)
    cols = list(raw.columns)
    inx  = [i for i, x in enumerate(cols) if x.strip() == "Capture Start Time"]
    if not inx:
        raise ValueError("'Capture Start Time' not found in CSV header.")
    return datetime.strptime(cols[inx[0] + 1].strip(), "%Y-%m-%d %I.%M.%S.%f %p")


def read_3_rigid_body_csv(_pth, rb_names=None):
    """
    Read a Motive CSV containing 3 rigid bodies and return 3 separate
    DataFrames, one per rigid body.

    Parameters
    ----------
    _pth     : str
        Path to the Motive-exported CSV file.
    rb_names : list of 3 str, optional
        Friendly labels for the rigid bodies in the order they appear in
        the file.  From your example file: ["tframe", "noark", "table"].
        Defaults to ["rb1", "rb2", "rb3"].

    Returns
    -------
    rb_dfs  : dict {label: DataFrame}
        Keys are the labels from rb_names.
        Each DataFrame contains:
            frame, seconds
            <label>_rot_x/y/z/w          ← quaternion rotation
            <label>_pos_x/y/z            ← position
            <label>_pos_err              ← mean marker error
            <label>_marker_m1_x/y/z      ← rigid body marker positions
            <label>_marker_m1_mq  …      ← marker quality
    st_time : datetime
        Capture start time.
    """
    if rb_names is None:
        rb_names = ["rb1", "rb2", "rb3"]
    # Accept 2 or 3 rigid bodies
    if len(rb_names) not in (2, 3):
        raise ValueError("rb_names must contain 2 or 3 names.")

    st_time = _parse_capture_start_time(_pth)

    # load everything after the 2-row metadata block as plain strings
    raw = pd.read_csv(_pth, skiprows=2, header=None, dtype=str)

    type_row  = raw.iloc[0].fillna("").str.strip().tolist()   # row 0
    name_row  = raw.iloc[1].fillna("").str.strip().tolist()   # row 1
    dtype_row = raw.iloc[3].fillna("").str.strip().tolist()   # row 3
    axis_row  = raw.iloc[4].fillna("").str.strip().tolist()   # row 4

    data    = raw.iloc[5:].reset_index(drop=True)             # row 5 onward
    n_cols  = len(type_row)

   # ── find the rigid body object names (in file order) ─────────────────────
    rb_object_names = []
    for t, n in zip(type_row, name_row):
        if t == "Rigid Body" and n and n not in rb_object_names:
            rb_object_names.append(n)

    n_expected = len(rb_names)
    if len(rb_object_names) < n_expected:
        raise ValueError(
            f"Expected {n_expected} rigid bodies, found "
            f"{len(rb_object_names)}: {rb_object_names}"
        )
    rb_object_names = rb_object_names[:n_expected]

    # ── build one DataFrame per rigid body ───────────────────────────────────
    rb_dfs = {}

    for label, obj_name in zip(rb_names, rb_object_names):

        col_indices = [0, 1]                          # frame, seconds always first
        col_labels  = ["frame", "seconds"]

        for ci in range(2, n_cols):
            t  = type_row[ci]
            n  = name_row[ci]
            dt = dtype_row[ci]
            ax = axis_row[ci].lower() if axis_row[ci] else ""

            # ── Rigid Body columns ────────────────────────────────────────────
            if t == "Rigid Body" and n == obj_name:
                if dt == "Rotation":
                    col_indices.append(ci)
                    col_labels.append(f"{label}_rot_{ax}")

                elif dt == "Position":
                    col_indices.append(ci)
                    col_labels.append(f"{label}_pos_{ax}")

                elif dt == "Mean Marker Error":
                    col_indices.append(ci)
                    col_labels.append(f"{label}_pos_err")

            # ── Rigid Body Marker columns ─────────────────────────────────────
            elif t == "Rigid Body Marker" and n.startswith(obj_name + ":"):
                marker_part = (
                    n.split(":")[1].lower().replace("marker", "").strip()
                )
                try:
                    m_idx = int(marker_part)
                except ValueError:
                    m_idx = marker_part

                if dt == "Position":
                    col_indices.append(ci)
                    col_labels.append(f"{label}_marker_m{m_idx}_{ax}")

                elif dt == "Marker Quality":
                    col_indices.append(ci)
                    col_labels.append(f"{label}_marker_m{m_idx}_mq")

        # slice and rename
        sub_df = data.iloc[:, col_indices].copy()
        sub_df.columns = col_labels
        sub_df = sub_df.reset_index(drop=True)
        for col in sub_df.columns:
            try:
                sub_df[col] = pd.to_numeric(sub_df[col])
            except (ValueError, TypeError):
                pass

        rb_dfs[label] = sub_df

    return rb_dfs, st_time


def read_markers_from_3rb_csv(_pth, rb_names=None):
    """
    Read the standalone Marker section of the CSV (separate from rigid body
    markers) and return a dict of DataFrames — one per rigid body.

    These are the unassigned / global markers exported alongside each body.

    Parameters
    ----------
    _pth     : path to the same CSV passed to read_3_rigid_body_csv
    rb_names : same 3-label list, defaults to ["rb1", "rb2", "rb3"]

    Returns
    -------
    marker_dfs : dict {label: DataFrame}
        Columns: frame, seconds, <label>_m1_x/y/z, <label>_m2_x/y/z, …
    st_time    : datetime
    """
    if rb_names is None:
        rb_names = ["rb1", "rb2", "rb3"]

    st_time = _parse_capture_start_time(_pth)

    raw       = pd.read_csv(_pth, skiprows=2, header=None, dtype=str)
    type_row  = raw.iloc[0].fillna("").str.strip().tolist()
    name_row  = raw.iloc[1].fillna("").str.strip().tolist()
    axis_row  = raw.iloc[4].fillna("").str.strip().tolist()
    data      = raw.iloc[5:].reset_index(drop=True)
    n_cols    = len(type_row)

    rb_object_names = []
    for t, n in zip(type_row, name_row):
        if t == "Rigid Body" and n and n not in rb_object_names:
            rb_object_names.append(n)
    rb_object_names = rb_object_names[:3]

    marker_dfs = {}

    for label, obj_name in zip(rb_names, rb_object_names):
        col_indices = [0, 1]
        col_labels  = ["frame", "seconds"]

        for ci in range(2, n_cols):
            t  = type_row[ci]
            n  = name_row[ci]
            ax = axis_row[ci].lower() if axis_row[ci] else ""

            if t == "Marker" and n.startswith(obj_name + ":"):
                marker_part = (
                    n.split(":")[1].lower().replace("marker", "").strip()
                )
                try:
                    m_idx = int(marker_part)
                except ValueError:
                    m_idx = marker_part
                col_indices.append(ci)
                col_labels.append(f"{label}_m{m_idx}_{ax}")

        sub_df = data.iloc[:, col_indices].copy()
        sub_df.columns = col_labels
        sub_df = sub_df.reset_index(drop=True)
        # sub_df = sub_df.apply(pd.to_numeric, errors="ignore")
        for col in sub_df.columns:
            try:
                sub_df[col] = pd.to_numeric(sub_df[col])
            except (ValueError, TypeError):
                pass
        marker_dfs[label] = sub_df

    return marker_dfs, st_time


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-BODY UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def add_datetime_col_3rb(rb_dfs, st_time):
    """
    Adds an absolute 'time' datetime column to each of the 3 rigid body
    DataFrames using the capture start time.
    """
    for label in rb_dfs:
        rb_dfs[label] = add_datetime_col(rb_dfs[label], st_time, "seconds")
    return rb_dfs


def trunkate_3_dfs(df_1, df_2, df_3, display_print=False):
    """
    Truncates 3 DataFrames to their shared overlapping time window.
    Each DataFrame must have a 'time' column.
    Returns (df_1, df_2, df_3).
    """
    df_1, df_2 = trunkate_dfs(df_1, df_2, display_print=display_print)
    df_1, df_3 = trunkate_dfs(df_1, df_3, display_print=display_print)
    df_2, df_3 = trunkate_dfs(df_2, df_3, display_print=display_print)
    return df_1, df_2, df_3


def interpolate_3rb_to_reference(rb_dfs, reference_df, col_names_map=None):
    """
    Resample all 3 rigid body DataFrames onto a common external timeline.

    rb_dfs        : dict {label: DataFrame} — must have a 'time' column
    reference_df  : DataFrame with 'time' column used as the target timeline
    col_names_map : optional dict {label: [col_names]} to interpolate.
                    If None, all numeric columns (except frame/seconds/time)
                    are interpolated automatically.

    Returns updated rb_dfs on the reference timeline.
    """
    result = {}
    for label, df in rb_dfs.items():
        cols = (
            col_names_map.get(label)
            if col_names_map
            else [
                c for c in df.columns
                if c not in ("frame", "seconds", "time")
                and pd.api.types.is_numeric_dtype(df[c])
            ]
        )
        result[label] = interpolate_target_df(df, reference_df, col_names=cols)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# NAMING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_rb_pos_cols(label):
    """
    Position column names for a rigid body.
    get_rb_pos_cols("tframe") → {"x": "tframe_pos_x", "y": ..., "z": ...}
    """
    return {ax: f"{label}_pos_{ax}" for ax in ("x", "y", "z")}


def get_rb_rot_cols(label):
    """
    Quaternion rotation column names for a rigid body.
    get_rb_rot_cols("tframe") → {"x": ..., "y": ..., "z": ..., "w": ...}
    """
    return {ax: f"{label}_rot_{ax}" for ax in ("x", "y", "z", "w")}


def get_rb_marker_name_3rb(label, val):
    """
    x/y/z column names for a rigid body marker.
    get_rb_marker_name_3rb("tframe", 1)
    → {"x": "tframe_marker_m1_x", "y": "tframe_marker_m1_y", "z": "tframe_marker_m1_z"}
    """
    return {ax: f"{label}_marker_m{val}_{ax}" for ax in ("x", "y", "z")}


def get_marker_name_3rb(label, val):
    """
    x/y/z column names for a standalone (unassigned) marker.
    get_marker_name_3rb("noark", 2)
    → {"x": "noark_m2_x", "y": "noark_m2_y", "z": "noark_m2_z"}
    """
    return {ax: f"{label}_m{val}_{ax}" for ax in ("x", "y", "z")}


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    FILE = "E:/Ragav/MS Bio Engineering/NOARK_backbone/mocap_data/table_frame.csv"

    # ── 1. Read all 3 rigid bodies ────────────────────────────────────────────
    # rb_names order must match the order in the CSV: tframe → noark → table
    rb_dfs, st_time = read_3_rigid_body_csv(
        FILE,
        rb_names=["noark", "table"],
    )

    # rb_dfs["tframe"] columns example:
    #   frame, seconds,
    #   tframe_rot_x, tframe_rot_y, tframe_rot_z, tframe_rot_w,
    #   tframe_pos_x, tframe_pos_y, tframe_pos_z, tframe_pos_err,
    #   tframe_marker_m1_x, tframe_marker_m1_y, tframe_marker_m1_z, tframe_marker_m1_mq,
    #   tframe_marker_m2_x … (4 markers total for tframe)
    #
    # rb_dfs["noark"]  → same pattern, noark_ prefix, 5 markers
    # rb_dfs["table"]  → same pattern, table_ prefix, 5 markers

    # ── 2. Add absolute datetime to each DataFrame ───────────────────────────
    rb_dfs = add_datetime_col_3rb(rb_dfs, st_time)

    rb_dfs["noark"], rb_dfs["table"] = trunkate_dfs(
        rb_dfs["noark"],
        rb_dfs["table"],
        display_print=True,
    )

    # ── 4. (Optional) Read standalone markers ────────────────────────────────
    marker_dfs, _ = read_markers_from_3rb_csv(
        FILE,
        rb_names=["tframe", "noark", "table"],
    )
    # marker_dfs["noark"] has columns: frame, seconds,
    #   noark_m1_x/y/z, noark_m2_x/y/z, noark_m3_x/y/z, noark_m4_x/y/z, noark_m5_x/y/z

    # ── 5. Use naming helpers ────────────────────────────────────────────────
    # print(get_rb_pos_cols("tframe"))
    # → {"x": "tframe_pos_x", "y": "tframe_pos_y", "z": "tframe_pos_z"}

    # print(get_rb_rot_cols("tframe"))
    # → {"x": "tframe_rot_x", "y": "tframe_rot_y", "z": "tframe_rot_z", "w": "tframe_rot_w"}

    # print(get_rb_marker_name_3rb("noark", 1))
    # → {"x": "noark_marker_m1_x", "y": "noark_marker_m1_y", "z": "noark_marker_m1_z"}

    # ── 6. Quick inspection ──────────────────────────────────────────────────
    for name, df in rb_dfs.items():
        print(f"\n── {name}  shape={df.shape} ──")
        print(df.columns.tolist())
        print(df.head(3).to_string())