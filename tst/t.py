import matplotlib.pyplot as plt
from matplotlib.widgets import Button,TextBox,CheckButtons,Slider,Cursor


a = plt.axes()
a.plot(range(134))
wcur = Cursor(a,useblit=True, color='red', linewidth=2)

plt.show()