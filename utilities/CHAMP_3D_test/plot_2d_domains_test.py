import pickle
from utilities import CHAMP_3D, plot_2d_domains
import matplotlib.pyplot as plt

G_intralayer, G_interlayer, layer_vec, champ_parts = pickle.load(open("CHAMP_3D_test.p", "rb"))
champ_parts = list(set(tuple(p) for p in champ_parts))
domains = CHAMP_3D(G_intralayer, G_interlayer, layer_vec, champ_parts, 0.0, 1.0, 0.0, 1.0)

plot_2d_domains(domains, [0, 1], [0, 1])
plt.savefig("our_CHAMP_test.png", dpi=300)
