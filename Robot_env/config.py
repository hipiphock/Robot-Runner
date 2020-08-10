from collections import namedtuple
import numpy as np

attr = namedtuple("Object", ['name', 'color'])

RL_Obj_List = {
    0:  attr('Background ',     (  0,   0,   0)),
    1:  attr('Drawer_light',    (  0,   0, 255)),
    2:  attr('Drawer_Dark',     (  0, 255,   0)),
    3:  attr('Bin_white',       (255,   0,   0)),
    4:  attr('Bin_stan',        (  0, 255, 255)),
    5:  attr('Holder_green',    (255,   0, 255)),
    6:  attr('Holder_black',    (255, 255,   0)),
    7:  attr('Keyboard_black',  (  0, 127, 255)),
    8:  attr('Keyboard_pink',   (127,   0, 255)),
    9:  attr('Cup_red',         (  0, 255, 127)),
    10: attr('Cup_pink',        (127, 255,   0)),
    11: attr('Box_small',       (255,   0, 127)),
    12: attr('Box_Big',         (255, 127,   0)),
    13: attr('Connecter_small', (30, 127,   90)),
    14: attr('Connecter_big',   (127, 127, 255)),
    15: attr('Milk',            (127, 255, 127)),
    16: attr('Yogurt',          (255, 127, 127)),
    17: attr('Eraser_small',    (127, 127, 127)),
    18: attr('Eraser_big',      ( 63, 127, 255)),
    19: attr('Usb_small',       ( 127, 63, 255)),
    20: attr('Usb_Big',         ( 63, 255, 127)),
    21: attr('Tape_black',      (127, 255,  63)),
    22: attr('Tape_white',      (255,  63, 127)),
    23: attr('Glue_pen',        (255, 127,  63)),
    24: attr('Glue_stick',      (  0,  63, 255)),
    25: attr('Stapler_pink',    ( 63,   0, 255)),
    26: attr('Stapler_stan',    (  0, 255,  63)),
    27: attr('Pen_namepen',     ( 63, 255,   0)),
    28: attr('Pen_marker',      (255,   0,  63)),
    29: attr('Usb_c',           (255,  63,   0)),
    30: attr('Usb_HDMI',        (127,  63,   127)),
}

# = 20191130 이전 RL캘리브레이션
# M_k2b = np.array([[0.0041381, 1.01268, -0.0129967, -0.5177],
#                   [1.00797, 0.00518543, 0.00592085, 0.215989],
#                   [0.0162992, -0.021624, -0.992748, 0.686926],
#                   [0, 0, 0, 1]])
# = 20191130 이전 RL캘리브레이션
M_k2b = np.array([[0.00982311, 1.00983, -0.0190632, -0.498649],
                  [1.0054, -0.00948272, 0.00791911, 0.237253],
                  [0.0131828, -0.0198063, -0.982348, 0.722444],
                  [0, 0, 0, 1]])
# 0.00982311, 1.00983, -0.0190632, -0.498649,
# 1.0054, -0.00948272, 0.00791911, 0.237253,
# 0.0131828, -0.0198063, -0.982348, 0.722444,