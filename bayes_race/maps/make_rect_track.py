import numpy as np
import csv
import shapely.geometry as shp
import matplotlib.pyplot as plt

WIDTH = 0.8  # [m]
Length = 6  # m, the length of the track along the centreline
Breadth = 7  # m, the breath of the track along the centreline
r_i = 3 * WIDTH / 4  # m, inner radius
r_o = r_i + WIDTH  # outer radius
r_c = (r_i + r_o)/2  # radius of the centre line

breadth = Breadth - 2*r_c  # breadth of the straights
length = Length - 2*r_c  # length of the straights

theta = np.pi/2  # rad, the angle of the corner
r_r = r_i + (r_o - r_i)/(1-np.cos(theta/2))  # radius of the geometric line

pts = 20  # number of points in the straights
length_arr = np.linspace(0, length, pts)
breath_arr = np.linspace(0, breadth, pts)
ones = np.ones(pts)

angles = np.linspace(0, np.pi / 2, 10)

#region Track boundary
seg1 = np.array([np.zeros(pts), length_arr])
arc1 = np.array([r_c * (1 - np.cos(angles)), length + r_c * np.sin(angles)])
seg2 = np.array([r_c + breath_arr, (length + r_c) * ones])
arc2 = np.array([breadth + r_c * (1 + np.sin(angles)), length + r_c * np.cos(angles)])
seg3 = np.array([np.zeros(pts) + 2 * r_c + breadth, length_arr[::-1]])
arc3 = np.array([breadth + r_c * (1 + np.cos(angles)), -r_c * np.sin(angles)])
seg4 = np.array([breath_arr[::-1] + r_c, -r_c * ones])
arc4 = np.array([r_c * (1 - np.sin(angles)), -r_c * np.cos(angles)])

centreline = np.hstack((seg1, arc1, seg2, arc2, seg3, arc3, seg4, arc4))
centreline[1] += r_c  # shifting the track up

track_xy = centreline.T
track_poly = shp.Polygon(track_xy)
track_xy_offset_in = track_poly.buffer(-WIDTH/2)
track_xy_offset_out = track_poly.buffer(WIDTH/2)
track_xy_offset_in_np = np.array(track_xy_offset_in.exterior).T
track_xy_offset_out_np = np.array(track_xy_offset_out.exterior).T
#endregion Track boundary

#region racing line
arc1 = np.array([-WIDTH / 2 + r_r * (1 - np.cos(angles)), length - (r_r - r_i) * np.sin(theta / 2) + r_r * np.sin(angles)])
arc2 = np.array([-WIDTH / 2 - 2 * (r_r-r_i) * np.sin(theta/2) + breadth + r_r * (1 + np.sin(angles)), -(r_r - r_i) * np.cos(theta / 2) + length + r_r * np.cos(angles)])
arc3 = np.array([-WIDTH / 2 - 2 * (r_r-r_i) * np.sin(theta/2) + breadth + r_r * (1 + np.cos(angles)), (r_r - r_i) * np.cos(theta / 2) - r_r * np.sin(angles)])
arc4 = np.array([-WIDTH / 2 + r_r * (1 - np.sin(angles)), (r_r - r_i) * np.cos(theta / 2) - r_r * np.cos(angles)])
raceline = np.hstack((arc1,arc2, arc3, arc4))
raceline[1] += r_c  # shifting the track up
#endregion raceline

fig, ax = plt.subplots()
ax.plot(*track_xy_offset_in_np, "k")
ax.plot(*track_xy_offset_out_np, 'k')
ax.plot(*track_xy[0], "go")
ax.plot(*track_xy_offset_out_np[[0, 1], 0], "ko")
ax.plot(*track_xy_offset_in_np[[0, 1], 1], "co")
ax.plot(*raceline, lw=3, c="c")
ax.plot(*(arc1[:,-1] + [0, r_c]), "ko")
ax.set_aspect('equal')
plt.show()

