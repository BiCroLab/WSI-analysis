from nd2reader import ND2Reader
import matplotlib.pyplot as plt
import sys
import seaborn as sns; sns.set(style='white', rc={'figure.figsize':(50,50)})


with ND2Reader(sys.argv[1]) as images:
  plt.axis('off')
  plt.imshow(images[0])
  plt.savefig('test.png',dpi=200)
  plt.close()


