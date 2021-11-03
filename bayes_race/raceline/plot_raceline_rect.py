"""	Plot optimal racing lines from saved results, for the rectangular track.
"""

__author__ = 'Danilo Martins'
__email__ = 'mrtdan014@myuct.ac.za'

from bayes_race.tracks import Rectangular
from bayes_race.params import F110
from plot_raceline import plot_raceline

# saved results 1
# savestr = '20211007203017'  # garbage line
# savestr = '20211007230356'  # garbage line
# savestr = '20211007234343'  # garbage line
# savestr = '20211008225852'  # good line on 6x4, 0.5m wide track, 1 iteration, F110
# savestr = '20211008234318'  # good line on 6x4, 0.8m wide track, 3 iterations, F110
# savestr = '20211009125025'  # good line on 6x4, 0.8m wide track, 3 iterations, F110
# savestr = '20211009182342'  # good line on 6x4, 0.8m wide track, 3 iterations, F110

# savestr = '20211013231001'  # ok line on rounded 6x4, 0.8m wide track, 1 iterations, F110
# savestr = '20211013233938'  # ok line on rounded 6x4, 0.8m wide track, 1 iterations, F110
# savestr = '20211013235614'  # good line on rounded 6x4, 0.8m wide track, 1 iterations, F110
savestr = '20211014001407'  # good line on rounded 4x6, 0.8m wide track, 1 iterations, F110
# savestr = '2021-10-14-17_17_09'  # good line on rounded 4x6, 0.8m wide track, 3 iterations, F110

track_params = {"length": 4, "breadth": 6, "width": 0.8, "r_corner": 0.3}
plot_raceline(Rectangular, track_params, 0, F110, savestr, save_results=False)

