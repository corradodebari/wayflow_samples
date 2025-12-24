# ----------------------------------------------------------------------
# WayFlow Code Example - Evaluate the SQLcl translation
#
# MIT License
# Copyright (c) 2025 Corrado De Bari
# ----------------------------------------------------------------------

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
import datetime as dt
import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence, Union

import re

import logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s"
)
# Disable httpx and httpcore noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

import oracledb
import pandas as pd

CFG = {}

Params = Union[Sequence[Any], Mapping[str, Any], None]


def _normalize(v: Any, float_tol: float) -> Any:
    """Normalize common Oracle-returned types into stable, comparable values."""
    if v is None or isinstance(v, (str, int, bool)):
        return v

    if isinstance(v, (datetime, date)):
        return v.isoformat()

    if isinstance(v, Decimal):
        return str(v)  # stable

    if isinstance(v, float):
        # optional bucketing for approximate float comparisons
        return round(v / float_tol) * float_tol if float_tol and float_tol > 0 else v

    if isinstance(v, (bytes, bytearray, memoryview)):
        b = bytes(v)
        return {"__bytes_sha256__": hashlib.sha256(b).hexdigest(), "__len__": len(b)}

    if isinstance(v, oracledb.LOB):
        # If you set oracledb.defaults.fetch_lobs = False, you typically won't see LOB objects here.
        data = v.read()
        if isinstance(data, str):
            b = data.encode("utf-8", errors="replace")
            return {"__clob_sha256__": hashlib.sha256(b).hexdigest(), "__len__": len(data)}
        else:
            data = bytes(data)
            return {"__blob_sha256__": hashlib.sha256(data).hexdigest(), "__len__": len(data)}

    # fallbacks for structured types
    if isinstance(v, (list, tuple)):
        return tuple(_normalize(x, float_tol) for x in v)
    if isinstance(v, dict):
        return {k: _normalize(v[k], float_tol) for k in sorted(v)}

    return str(v)


def _exec_and_get_colnames(conn: oracledb.Connection, sql: str):
    if not isinstance(sql, str):
        raise TypeError(f"SQL must be a string. Got {type(sql)}. (Did you leave a trailing comma?)")
    cur = conn.cursor()
    cur.execute(sql)   # execute as-is (no params)
    colnames = [d[0] for d in cur.description] if cur.description else []
    return cur, colnames


def same_sql_results_oracle_by_column(
    conn: oracledb.Connection,
    sql_reference: str,       
    sql_actual: str,
    *,
    ignore_order: bool = True,
    consider_duplicates: bool = True,
    float_tol: float = 0.0,
    arraysize: int = 1000,
    require_same_columns: bool = True, 
          #True:  All the columns must be the same with the same name. 
          #False: All the columns in sql_reference must be in sql_actual column list. 
          #      The order of columns is not considered, The name is not considered: only the column content
) -> bool:
    cur_a, cols_a = _exec_and_get_colnames(conn, sql_reference)
    cur_b, cols_b = _exec_and_get_colnames(conn, sql_actual)

    logger.info(f"A: {cols_a} ")
    logger.info(f"B: {cols_b} ")

    try:
        cur_a.arraysize = arraysize
        cur_b.arraysize = arraysize

        # get all data
        all_data_a = []
        while True:
            batch = cur_a.fetchmany(arraysize)
            if not batch:
                break
            all_data_a.extend(batch)

        all_data_b = []
        while True:
            batch = cur_b.fetchmany(arraysize)
            if not batch:
                break
            all_data_b.extend(batch)

        df_a = pd.DataFrame(all_data_a, columns=cols_a)
        df_b = pd.DataFrame(all_data_b, columns=cols_b)

        if require_same_columns:
            if set(cols_a) != set(cols_b):
                return False
            # normalize for float_tol
            def normalize_df(df):
                return df.apply(lambda x: x.map(lambda y: _normalize(y, float_tol)))
            df_a_norm = normalize_df(df_a)
            df_b_norm = normalize_df(df_b)
            # then do row comparison using pandas
            if not ignore_order:
                # sort columns by name for consistent order
                df_a_norm = df_a_norm.sort_index(axis=1)
                df_b_norm = df_b_norm.sort_index(axis=1)
                return df_a_norm.equals(df_b_norm)
            else:
                # sort columns by name first
                df_a_norm = df_a_norm.sort_index(axis=1)
                df_b_norm = df_b_norm.sort_index(axis=1)
                df_a_sorted = df_a_norm.sort_values(by=df_a_norm.columns.tolist()).reset_index(drop=True)
                df_b_sorted = df_b_norm.sort_values(by=df_b_norm.columns.tolist()).reset_index(drop=True)
                if consider_duplicates:
                    return df_a_sorted.equals(df_b_sorted)
                else:
                    df_a_unique = df_a_sorted.drop_duplicates()
                    df_b_unique = df_b_sorted.drop_duplicates()
                    return df_a_unique.equals(df_b_unique)

        # column matching: check if all columns of a (the reference columns) have matching in b (the actual translated), b can have extra
        col_data_a = []
        for col in df_a.columns:
            series = df_a[col].copy()
            if ignore_order:
                series = series.sort_values()
            if not consider_duplicates:
                series = series.drop_duplicates()
            normalized = [_normalize(v, float_tol) for v in series.values]
            col_data_a.append(tuple(normalized))

        col_data_b = []
        for col in df_b.columns:
            series = df_b[col].copy()
            if ignore_order:
                series = series.sort_values()
            if not consider_duplicates:
                series = series.drop_duplicates()
            normalized = [_normalize(v, float_tol) for v in series.values]
            col_data_b.append(tuple(normalized))

        # check if all col_data_a are in col_data_b
        counter_b = Counter(col_data_b)
        for col in col_data_a:
            if counter_b[col] == 0:
                return False
            counter_b[col] -= 1
        return True

    finally:
        cur_a.close()
        cur_b.close()


import datetime as dt
import decimal
import hashlib
import math
from collections import Counter, defaultdict

def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def make_normalizer(float_factor=None):
    """
    float_factor:
      - None   -> exact float matching (float.hex)
      - number -> compare floats by int(round(v * float_factor))
                 e.g. 100 => 2 decimal places
    """
    def norm(v):
        if v is None:
            return ("NULL",)

        # LOBs (python-oracledb LOB objects have .read())
        if hasattr(v, "read") and callable(getattr(v, "read")) and not isinstance(v, (str, bytes, bytearray, memoryview)):
            data = v.read()
            if isinstance(data, str):
                return ("CLOB_SHA256", _sha256_hex(data.encode("utf-8")))
            return ("BLOB_SHA256", _sha256_hex(bytes(data)))

        if isinstance(v, decimal.Decimal):
            return ("DEC", str(v.normalize()))

        if isinstance(v, bool):
            return ("BOOL", int(v))

        if isinstance(v, int):
            return ("INT", v)

        if isinstance(v, float):
            if math.isnan(v):
                return ("FLOAT", "NaN")
            if math.isinf(v):
                return ("FLOAT", "+Inf" if v > 0 else "-Inf")
            if v == 0.0:
                v = 0.0  # collapse -0.0
            if float_factor is None:
                return ("FLOAT", float.hex(v))
            return ("FLOAT_SCALED", float_factor, int(round(v * float_factor)))

        if isinstance(v, dt.datetime):
            if v.tzinfo is None:
                return ("DATETIME", v.isoformat(timespec="microseconds"))
            return ("DATETIME_UTC", v.astimezone(dt.timezone.utc).isoformat(timespec="microseconds"))

        if isinstance(v, dt.date) and not isinstance(v, dt.datetime):
            return ("DATE", v.isoformat())

        if isinstance(v, dt.time):
            return ("TIME", v.isoformat(timespec="microseconds"))

        if isinstance(v, dt.timedelta):
            return ("INTERVAL_US", int(v.total_seconds() * 1_000_000))

        if isinstance(v, (bytes, bytearray, memoryview)):
            return ("RAW_SHA256", _sha256_hex(bytes(v)))

        if isinstance(v, str):
            return ("STR", v)

        return (type(v).__name__, repr(v))

    return norm


def equivalent_allow_rowcol_perm(
    conn,
    sql_reference="",
    sql_actual="",
    ignore_order=True,
    require_same_columns_names=False,
    require_same_columns=False,
    float_factor=None,
    arraysize = 1000
):
    """
    A_rows, B_rows: rows from python-oracledb cursor.fetchall() (list[tuple])

    ignore_order:
      - False: only column permutation (rows fixed)
      - True : column permutation + row permutation allowed

    require_same_columns_names:
      - True: require same column names (as multiset) AND same #columns

    require_same_columns:
      - True: B must have same #columns as A (no pruning/subset selection)
      - False: B may have extra columns; try to match A inside B
    """
    cursorA = conn.cursor()
    cursorA.execute(sql_reference)
    cursorB = conn.cursor()
    cursorB.execute(sql_actual)
 
    # rows
    A_rows = cursorA.fetchmany(arraysize)
    B_rows = cursorB.fetchmany(arraysize)

    # names (Oracle usually uppercases them)

    A_colnames = [d[0] for d in cursorA.description]
    B_colnames = [d[0] for d in cursorB.description]
    
    norm = make_normalizer(float_factor=float_factor)

    A_rows = [tuple(r) for r in A_rows]
    B_rows = [tuple(r) for r in B_rows]

    if not A_rows and not B_rows:
        return True
    if not A_rows or not B_rows:
        return False

    ra, ca = len(A_rows), len(A_rows[0])
    rb, cb = len(B_rows), len(B_rows[0])

    if any(len(r) != ca for r in A_rows) or any(len(r) != cb for r in B_rows):
        raise ValueError("Inconsistent row lengths in A or B.")
    if ra != rb:
        return False
    if require_same_columns and ca != cb:
        return False
    if ca > cb:
        return False

    # ---- column names check (if requested)
    if require_same_columns_names:
        if A_colnames is None or B_colnames is None:
            raise ValueError("require_same_columns_names=True needs A_colnames and B_colnames.")
        if len(A_colnames) != len(B_colnames):
            return False
        # since we allow column permutation, compare as multisets (case/space normalized)
        def nname(x): return str(x).strip().upper()
        if Counter(map(nname, A_colnames)) != Counter(map(nname, B_colnames)):
            return False
        # same names implies same number of columns
        if ca != cb:
            return False

    # Convert to columns
    A_cols = list(zip(*A_rows))
    B_cols = list(zip(*B_rows))

    # Helper: column vector with rows fixed
    def col_vector_fixed(col):
        return tuple(norm(x) for x in col)

    # Helper: row-order invariant fingerprint (multiset of values in column)
    def col_fingerprint_row_invariant(col):
        c = Counter(norm(x) for x in col)
        return tuple(sorted(c.items()))

    # ----------------------------
    # Case 1) rows fixed (ignore_order=False): only column permutation
    # ----------------------------
    if not ignore_order:
        A_vecs = Counter(col_vector_fixed(A_cols[j]) for j in range(ca))

        if require_same_columns:
            B_vecs = Counter(col_vector_fixed(B_cols[j]) for j in range(cb))
            return A_vecs == B_vecs

        # allow B extra cols: A columns must be contained in B columns
        B_vecs = Counter(col_vector_fixed(B_cols[j]) for j in range(cb))
        return all(B_vecs[v] >= k for v, k in A_vecs.items())

    # ----------------------------
    # Case 2) rows permutable (ignore_order=True): column permutation + row permutation
    # ----------------------------

    # Optional pruning: if B has extra cols and pruning is allowed, keep only what A needs
    if not require_same_columns:
        need = Counter(col_fingerprint_row_invariant(A_cols[j]) for j in range(ca))
        buckets = defaultdict(list)
        for j in range(cb):
            buckets[col_fingerprint_row_invariant(B_cols[j])].append(j)

        keep = []
        for fp, k in need.items():
            if len(buckets[fp]) < k:
                return False
            keep.extend(buckets[fp][:k])

        B_cols = [B_cols[j] for j in keep]
        cb = len(B_cols)

    # Now A and B_cols must have same number of columns to match
    if ca != cb:
        return False

    # Build candidates per A col based on row-invariant fingerprint
    A_fp = [col_fingerprint_row_invariant(A_cols[i]) for i in range(ca)]
    B_fp = [col_fingerprint_row_invariant(B_cols[j]) for j in range(ca)]

    fp_to_B = defaultdict(list)
    for j, fp in enumerate(B_fp):
        fp_to_B[fp].append(j)

    candidates = []
    for i, fp in enumerate(A_fp):
        cand = fp_to_B.get(fp, [])
        if not cand:
            return False
        candidates.append((i, cand))

    # Fewest candidates first (big speedup)
    candidates.sort(key=lambda t: len(t[1]))

    used = set()
    sigA = [()] * ra
    sigB = [()] * ra

    def dfs(pos, sigA, sigB):
        if pos == ca:
            return True
        i, cand_js = candidates[pos]
        for j in cand_js:
            if j in used:
                continue

            new_sigA = [sigA[r] + (norm(A_cols[i][r]),) for r in range(ra)]
            new_sigB = [sigB[r] + (norm(B_cols[j][r]),) for r in range(ra)]

            # A row permutation exists for chosen columns iff row-tuples match as a multiset
            if Counter(new_sigA) != Counter(new_sigB):
                continue

            used.add(j)
            if dfs(pos + 1, new_sigA, new_sigB):
                return True
            used.remove(j)
        return False

    return dfs(0, sigA, sigB)



DB_CFG = {
    "dsn": "localhost:1521/FREEPDB1",
    "user": "SH",
    "password": "Welcome_12345", #put [SCHEMA_PASSWORD]
}

if __name__ == "__main__":
    
    _ignore_order = True
    _require_same_columns = False
    _require_same_columns_names = False
    _consider_duplicates = False
    _float_tol = 100
    _arraysize = 1000

    
    # Recommended for stable comparisons (esp. LOBs and NUMBER precision):
    oracledb.defaults.fetch_lobs = False       # fetch CLOB/BLOB as str/bytes  
    oracledb.defaults.fetch_decimals = True    # fetch NUMBER as Decimal  

    conn = oracledb.connect(user=DB_CFG["user"], password=DB_CFG["password"], dsn=DB_CFG["dsn"])

    nl_q1="give me the top ten selling in the month of march and which sales has done"

    #Example: less columns, different name/same content, order required
    q1 = "SELECT PROD_ID as ID,UNIT_COST \nFROM SH.COSTS\nWHERE TIME_ID > TO_DATE('2020-01-01', 'YYYY-MM-DD') order by ID"
    q2 = "SELECT PROD_ID, UNIT_PRICE, UNIT_COST \nFROM SH.COSTS\nWHERE TIME_ID > TO_DATE('2020-01-01', 'YYYY-MM-DD') order by PROD_ID"

    same = equivalent_allow_rowcol_perm(
                    conn, 
                    sql_reference  = q1,
                    sql_actual = q2 , 
                    ignore_order=_ignore_order,                  # allow row permutation
                    require_same_columns_names=_require_same_columns_names,     # fail if column names differ
                    require_same_columns=_require_same_columns,          # allow B extra columns (prune)
                    float_factor=_float_tol,                   # compare floats at 2 decimals
                    )
    
    print(f"Same? {same}")

    total = 0
    positive = 0
    with open('cache/dataflow_cache_step_step3_q_a.jsonl', 'r') as f,  open('cache/dataflow_cache_step_step3_q_a_test.jsonl', 'w') as outfile:
        for line in f:
            json_str = line.strip()  # Remove any trailing newline
            if json_str:  # Skip empty lines if any
                data = json.loads(json_str)
                # get reference sql
                sql_ref = data.get('SQL')
                if sql_ref.endswith(';'):
                    sql_ref=sql_ref[:-1]
                    pattern = r'\bORDER\s+BY\b'
                    _ignore_order=not bool(re.search(pattern, sql_ref, re.IGNORECASE))
                question = data.get('question')

                # get nl2sql generated
                sql_actual=data.get('NL2SQL') 
                if sql_actual.endswith(';'):
                    sql_actual=sql_actual[:-1]

                logger.info(f"Question:\n{question}")
                logger.info(f"SQL:\n{sql_ref}")
                logger.info(f"NL2SQL:\n{sql_actual}")

                same = equivalent_allow_rowcol_perm(
                    conn, 
                    sql_reference  = sql_ref,
                    sql_actual = sql_actual , 
                    ignore_order=_ignore_order,                  # allow row permutation
                    require_same_columns_names=_require_same_columns_names,     # fail if column names differ
                    require_same_columns=_require_same_columns,          # allow B extra columns (prune)
                    float_factor=_float_tol                   # compare floats at 2 decimals
                    )
                
                data['test'] = same
                outfile.write(json.dumps(data) + '\n')
                logger.info(f"Result test: {same}")
                total +=1
                if (same):
                    positive +=1
    logger.info(f"\n\n--------------------------\nFinal score : {int(positive/total*100)}%\n--------------------------")


