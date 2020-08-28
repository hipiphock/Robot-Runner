from collections import namedtuple
import numpy as np

attr = namedtuple("Object", ['name', 'color'])

RL_Obj_List = {
    0:  attr('Background ',         (  0,   0,   0)),
    1:  attr('Drawer_light',        (  0,   0, 255)),
    2:  attr('Drawer_Dark',         (  0, 255,   0)),
    3:  attr('Bin_white',           (255,   0,   0)),
    4:  attr('Bin_stan',            (  0, 255, 255)),
    5:  attr('Holder_green',        (255,   0, 255)),
    6:  attr('Holder_black',        (255, 255,   0)),
    7:  attr('Keyboard_black',      (  0, 127, 255)),
    8:  attr('Keyboard_pink',       (127,   0, 255)),
    9:  attr('Cup_red',             (  0, 255, 127)),
    10: attr('Cup_pink',            (127, 255,   0)),
    11: attr('Box_small',           (255,   0, 127)),
    12: attr('Box_Big',             (255, 127,   0)),
    13: attr('Connecter_small',     ( 30, 127,  90)),
    14: attr('Connecter_big',       (127, 127, 255)),
    15: attr('Milk',                (127, 255, 127)),
    16: attr('Yogurt',              (255, 127, 127)),
    17: attr('Eraser_small',        (127, 127, 127)),
    18: attr('Eraser_big',          ( 63, 127, 255)),
    19: attr('Usb_small',           (127,  63, 255)),
    20: attr('Usb_Big',             ( 63, 255, 127)),
    21: attr('Tape_black',          (127, 255,  63)),
    22: attr('Tape_white',          (255,  63, 127)),
    23: attr('Glue_pen',            (255, 127,  63)),
    24: attr('Glue_stick',          (  0,  63, 255)),
    25: attr('Stapler_pink',        ( 63,   0, 255)),
    26: attr('Stapler_stan',        (  0, 255,  63)),
    27: attr('Pen_namepen',         ( 63, 255,   0)),
    28: attr('Pen_marker',          (255,   0,  63)),
    29: attr('Usb_c',               (255,  63,   0)),
    30: attr('Usb_HDMI',            (127,  63, 127)),
    31: attr('silver_namepen',      (127, 255,  63)),
    32: attr('black_marker',        (255,  63, 127)),
    33: attr('blue_marker',         (255, 127,  63)),
    34: attr('black_file_holder',   (  0,  63, 255)),
    35: attr('pink_file_holder',    ( 63,   0, 255)),
    36: attr('purple_book',         (  0, 255,  63)),
    37: attr('green_desk_cleaner',  ( 63, 255,   0)),
    38: attr('apricot_bottle',      (255,   0,  63)),
    39: attr('grey_bottle',         (255,  63,   0)),
    40: attr('white_book',          (127,  63, 127)),
    41: attr('skyblue_desk_cleaner',(127,  63, 127)),


}

# 20200813 RL Calibration
M_k2b = np.array([[0.0131835, 1.00082, 0.00988999, -0.518463],
                  [1.01042, -0.00466307, -0.0585077, 0.328917],
                  [0.0234888, -0.00175762, -0.996845, 0.718622],
                  [0, 0, 0, 1]])