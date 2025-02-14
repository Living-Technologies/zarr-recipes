#!/usr/bin/env python3

import ltzarr
import sys
import numpy

tags = ["label_id",	"anchor_x",	"anchor_y",	"anchor_z",	"bb_min_x",	"bb_min_y",	"bb_min_z",	"bb_max_x",	"bb_max_y",	"bb_max_z",	"n_pixels",	"timepoint",	"track_id" ]
multiscales = ltzarr.loadZarr( sys.argv[1] )

def extractRows(volume):
    values = numpy.unique(volume)
    ret = []
    for v in values:
        if v == 0:
            continue
        row = [0]*len(tags)
        row[0] = v
        
        ret.append(row)
    return ret
#Should only be one.
for ms in multiscales:
    rows = []
    for tp in range(ms.getNTimePoints()):
        lbl_volume = ms.getVolume(0, 0)[0]
        ers = extractRows( lbl_volume )
        t0 = len(rows)
        for i, row in enumerate(ers):
            tid = t0 + i
            row[-1] = tid
            row[-2] = tp
            rows.append( row )
        
    with open("testing.tsv", 'w') as lbl_file:
        lbl_file.write("%s\n"%"\t".join( tags ) )
        for row in rows:
            lbl_file.write("%s\n"%"\t".join("%s"%c for c in row))
        
    
    

