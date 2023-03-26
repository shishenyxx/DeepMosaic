import numpy as np

MAX_DP = 500
WIDTH = 300

def strand_to_index(reverse=False):
    if reverse:
        return 100
    else:
        return 200

def base_to_index(base):
    if base == "A":
        return 50
    elif base == "C":
        return 100
    elif base == "G":
        return 150
    elif base == "T":
        return 200
    elif base == "N":
        return 250



#image with 3 channel
def paint_canvas(reads, pos):
    canvas = np.zeros([MAX_DP, WIDTH, 3], dtype=np.uint8)
    pos = pos - 1
    start_pos = pos - WIDTH/2
    end_pos = pos + WIDTH/2
    for i, read in enumerate(reads):
        if read.reference_start < start_pos:
            for pos in read.get_reference_positions():
                if pos >= start_pos:
                    start = pos
                    break
            offset = int(start-start_pos)
        else:
            start = read.reference_start
            offset = int(read.reference_start - start_pos)
        read_sequence = read.query_sequence
        qualities = read.query_qualities
        ref_positions = read.get_reference_positions(full_length=len(read_sequence))
        strand_value = strand_to_index(read.is_reverse)
        for j, pos in enumerate(ref_positions):
            if pos==None or pos < start:
                continue
            canvas_index = pos-start+offset
            if canvas_index >= WIDTH:
                break
            base = read_sequence[j]
            canvas[i, canvas_index, 0] = base_to_index(base)
            canvas[i, canvas_index, 1] = qualities[j]
            canvas[i, canvas_index, 2] = strand_value
    return canvas
