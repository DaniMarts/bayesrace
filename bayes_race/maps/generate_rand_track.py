""" Generates random tracks.
    Adapted from https://gym.openai.com/envs/CarRacing-v0
    
"""

import cv2
import sys, math
import numpy as np
import shapely.geometry as shp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


NUM_MAPS = 20
WIDTH = 5.0

def create_track():
    CHECKPOINTS = 12
    SCALE = 15.0
    TRACK_RAD = 350/SCALE
    TRACK_DETAIL_STEP = 21/SCALE
    TRACK_TURN_RATE = 0.31

    start_alpha = 0.

    # Create checkpoints
    checkpoints = []
    for c in range(CHECKPOINTS):
        alpha = 2*math.pi*c/CHECKPOINTS + np.random.uniform(0, 2*math.pi*1/CHECKPOINTS)
        rad = np.random.uniform(TRACK_RAD/3, TRACK_RAD)
        if c==0:
            alpha = 0
            rad = 1.5*TRACK_RAD
        if c==CHECKPOINTS-1:
            alpha = 2*math.pi*c/CHECKPOINTS
            start_alpha = 2*math.pi*(-0.5)/CHECKPOINTS
            rad = 1.5*TRACK_RAD
        checkpoints.append( (alpha, rad*math.cos(alpha), rad*math.sin(alpha)) )
    road = []

    # Go from one checkpoint to another to create track
    x, y, beta = 1.5*TRACK_RAD, 0, 0
    dest_i = 0
    laps = 0
    track = []
    no_freeze = 2500
    visited_other_side = False
    while True:
        alpha = math.atan2(y, x)
        if visited_other_side and alpha > 0:
            laps += 1
            visited_other_side = False
        if alpha < 0:
            visited_other_side = True
            alpha += 2*math.pi
        while True:
            failed = True
            while True:
                dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                if alpha <= dest_alpha:
                    failed = False
                    break
                dest_i += 1
                if dest_i % len(checkpoints) == 0:
                    break
            if not failed:
                break
            alpha -= 2*math.pi
            continue
        r1x = math.cos(beta)
        r1y = math.sin(beta)
        p1x = -r1y
        p1y = r1x
        dest_dx = dest_x - x
        dest_dy = dest_y - y
        proj = r1x*dest_dx + r1y*dest_dy
        while beta - alpha >  1.5*math.pi:
             beta -= 2*math.pi
        while beta - alpha < -1.5*math.pi:
             beta += 2*math.pi
        prev_beta = beta
        proj *= SCALE
        if proj >  0.3:
             beta -= min(TRACK_TURN_RATE, abs(0.001*proj))
        if proj < -0.3:
             beta += min(TRACK_TURN_RATE, abs(0.001*proj))
        x += p1x*TRACK_DETAIL_STEP
        y += p1y*TRACK_DETAIL_STEP
        track.append( (alpha,prev_beta*0.5 + beta*0.5,x,y) )
        if laps > 4:
             break
        no_freeze -= 1
        if no_freeze==0:
             break

    # Find closed loop
    i1, i2 = -1, -1
    i = len(track)
    while True:
        i -= 1
        if i==0:
            return False
        pass_through_start = track[i][0] > start_alpha and track[i-1][0] <= start_alpha
        if pass_through_start and i2==-1:
            i2 = i
        elif pass_through_start and i1==-1:
            i1 = i
            break
    print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2-i1))
    assert i1!=-1
    assert i2!=-1

    track = track[i1:i2-1]
    first_beta = track[0][1]
    first_perp_x = math.cos(first_beta)
    first_perp_y = math.sin(first_beta)

    # Length of perpendicular jump to put together head and tail
    well_glued_together = np.sqrt(
        np.square( first_perp_x*(track[0][2] - track[-1][2]) ) +
        np.square( first_perp_y*(track[0][3] - track[-1][3]) ))
    if well_glued_together > TRACK_DETAIL_STEP:
        return False

    # post processing, converting to numpy, finding exterior and interior walls
    track_xy = [(x, y) for (a1, b1, x, y) in track]
    track_xy = np.asarray(track_xy)
    track_poly = shp.Polygon(track_xy)
    track_xy_offset_in = track_poly.buffer(-WIDTH)
    track_xy_offset_out = track_poly.buffer(WIDTH)
    track_xy_offset_in_np = np.array(track_xy_offset_in.exterior)
    track_xy_offset_out_np = np.array(track_xy_offset_out.exterior)
    # returning an array of the centreline waypoints, interior and exterior wall points
    return track_xy, track_xy_offset_in_np, track_xy_offset_out_np


def convert_track(track, track_int, track_ext, iter):
    """
    params:
        track: np.array representing track centreline points
        track_int: np.array representing inner wall points
        track_ext: np.array representing outer wall points
        iter: a number passes by a loop, to look for the right map
    """
    # converts track to image and saves the centerline as waypoints
    fig, ax = plt.subplots()
    # fig.set_size_inches(20, 20)
    ax.plot(*track_int.T, color='gray', linewidth=3)  # plotting the inside
    ax.plot(*track_ext.T, color='black', linewidth=3)  # plotting the outside
    plt.tight_layout()
    ax.set_aspect('equal')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    # ax.set_xlim(-180, 300)
    # ax.set_ylim(-300, 300)
    # plt.axis('off')
    plt.show()
    # plt.savefig('maps/map' + str(iter) + '.png', dpi=80)

    map_width, map_height = fig.canvas.get_width_height()
    print('map size: ', map_width, map_height)

    # transform the track center line into pixel coordinates
    xy_pixels = ax.transData.transform(track)
    origin_x_pix = xy_pixels[0, 0]
    origin_y_pix = xy_pixels[0, 1]

    xy_pixels = xy_pixels - np.array([[origin_x_pix, origin_y_pix]])

    map_origin_x = -origin_x_pix*0.05
    map_origin_y = -origin_y_pix*0.05

    # create yaml file
    # yaml = open('maps/map' + str(iter) + '.yaml', 'w')
    # yaml.write('image: map' + str(iter) + '.pgm\n')
    # yaml.write('resolution: 0.050000\n')
    # yaml.write('origin: [' + str(map_origin_x) + ',' + str(map_origin_y) + ', 0.000000]\n')
    # yaml.write('negate: 0\noccupied_thresh: 0.45\nfree_thresh: 0.196')
    # yaml.close()
    plt.close()

    # saving track centerline as a csv in ros coords
    waypoints_csv = open('waypoints/map' + str(iter) + '.csv', 'w')
    for row in xy_pixels:
        waypoints_csv.write(str(0.05*row[0]) + ', ' + str(0.05*row[1]) + '\n')
    waypoints_csv.close()

    # convert image using cv2
    # cv_img = cv2.imread('maps/map' + str(iter) + '.png')
    # cv2.imwrite('maps/map' + str(iter) + '.pgm', cv_img)


if __name__ == '__main__':
    for i in range(NUM_MAPS):
        try:
            track, track_int, track_ext = create_track()
        except:
            print('Random generator failed, retrying')
            continue
        convert_track(track, track_int, track_ext, i)
