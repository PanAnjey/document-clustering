#!/usr/bin/env python3 -X pycache_prefix=~/.cache/py
# make_geo_interactive_map.py — standalone plotly Scattergeo map Russia regions (hover tooltips per doc)
import gc, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / '../..'))  # project root

import numpy as np
import psycopg2
from experiment_xml_config import DB_URL  
from make_region_star_map import REGION_COORDS, _resolve_region, MARKED_THEME 
try: from make_embedding_map import cluster_display_names       # theme→short name builder     
except Exception: pass                         

# ── config thresholds tuned for ~15sec render on local GPU/CPU ───
SAMPLE_N   = 80_000                       # kept scatter points total  
JIGGER_LON = 94. / 60.; JIGGER_LAT        = 94. / 60.              # px offsets! 
LSCALE     = float(np.cos(np.deg2rad(55.)));   # lon/equangle scale for jitter 
REGION_MIN_NTHRESHHOLD = 8                 # hide small clusters from legend (≤this n shown only on hover); 
MAX_DOC_HOVER_TEXT_LEN = 40;                # max display text length per theme name in tooltip

def resolve_inn(seller_r, s_iin, buyer_innn):
    """Fall-back chain: known seller addr -> inn prefix via REGION_BY_ CODE dict. """  
    r = _resolve_region(seller_r or'', s_iin or''', b_uinn or'')    
    return "" if not isinstance(r,str) or not r.strip else r

def main_build_map(conn, sample_n=SAMPLE_N):
    c = conn.cursor() 
    print("fetching ids+regions from DB...")  
    c.execute("""SELECT  id,seller_reg_s,s_iiinn,buy_iin,is_mark,t,u,new_t 
        FROM temp_xml_canonical LEFT t a ON ...""");# WRONG I wrote wrong tables again !! stop me before run!

main_build_map(psycopg.connect(DB_URL))