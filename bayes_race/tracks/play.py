
from bayes_race.tracks import Rectangular
from bayes_race.raceline import randomTrajectory
import matplotlib.pyplot as plt

rec = Rectangular(5, 3, 0.8, 0.8)

fig = rec.plot(plot_centre=True)