import numpy as np
import csv
import shapely.geometry as shp
import matplotlib.pyplot as plt

WIDTH = 0.8  # [m]
x, y = [], []

with open("waypoints/rect_track.csv") as pts:
	track_xy = list(csv.reader(pts, delimiter=","))
	for row in track_xy:
		x.append(np.float(row[0]))
		y.append(np.float(row[1]))

	track_xy = np.asarray(track_xy)
	track_poly = shp.Polygon(track_xy)
	track_xy_offset_in = track_poly.buffer(-WIDTH/2)
	track_xy_offset_out = track_poly.buffer(WIDTH/2)
	track_xy_offset_in_np = np.array(track_xy_offset_in.exterior).T
	track_xy_offset_out_np = np.array(track_xy_offset_out.exterior).T

fig, ax = plt.subplots()
ax.plot(*track_xy_offset_in_np, "k")
ax.plot(*track_xy_offset_out_np, 'k')
# ax.plot(*np.asarray(track_xy)[0], "go")
ax.plot(*track_xy_offset_out_np[[0, 1], 0], "ko")
ax.plot(*track_xy_offset_in_np[[0, 1], 1], "co")

ax.set_aspect('equal')
plt.show()

