
from bayes_race.tracks import Rectangular
from bayes_race.raceline import randomTrajectory
import matplotlib.pyplot as plt

rec = Rectangular(7, 15, 2, 1.5)

fig = rec.plot(plot_centre=False)
plt.show()
