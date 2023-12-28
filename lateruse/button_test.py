import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import functools

spy = yf.download('SPY', period='max', interval='1d')
df = pd.DataFrame(spy)

fig, ax = plt.subplots()
ax.plot(df['Close'])

plt.subplots_adjust(bottom=0.2)

class UI():

    def all_buttons(self, x, event):
        ax.cla()
        if x =='max':
            ax.plot(df['Close'])
        else:
            m = x[1:2]
            weight = 1 if m == 'd' else 7 if m == 'w' else 30 if m == 'm' else 252
            t = int(x[0:1]) * weight
            ax.plot(df['Close'][-int(t):])
        
        self.style()
        fig.canvas.draw()

    def style(self):
        fig.set_facecolor('black')
        ax.set_facecolor('black')
        ax.set_title('SPY Close over time', color='white')
        ax.set_ylabel('Close Price', color='white')
        ax.set_xlabel('Date', color='white')
        ax.tick_params(axis='both', color='white')
        ax.grid(True, color='white', linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color('white')
        for i in range(len(instances)):
            bpos[i].label.set_color('green')
            bpos[i].label.set_fontsize(16)
            bpos[i].color = 'black'
            bpos[i].hovercolor = 'black'
            bpos[i].label.set_fontfamily('Arial')
        fig.canvas.draw()

chartUI = UI()
place_holders = {}
bpos = {}
instances = ['1d', '1w', '1m', '3m', '6m', '1y', '5y', 'max']

for i in range(len(instances)):
    place_holders[i] = plt.axes([0.1 + 0.11 * i, 0.05, 0.1, 0.05])
    bpos[i] = Button(place_holders[i], str(instances[i]))
    bpos[i].on_clicked(functools.partial(chartUI.all_buttons, instances[i]))

chartUI.style()
plt.show()


