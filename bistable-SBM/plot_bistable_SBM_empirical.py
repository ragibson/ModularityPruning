import matplotlib.pyplot as plt

output = open("SBM_constant_probs.out", "r").read()

run = output.split("\n\n")[0]
lines = run.split("\n")
N = float(lines[0])
p2s = eval(lines[1])
c2s = eval(lines[2])
c3s = eval(lines[3])
cboths = eval(lines[4])

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.scatter(p2s, c2s, label="Probability of $K=2$ stability", s=20)
plt.scatter(p2s, c3s, label="Probability of $K=3$ stability", s=20)
plt.plot(p2s, cboths, label="Probability of bistability", alpha=0.75, color="C2")
plt.xlabel(r"$\delta$", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.title("Simulation of Example Bistable SBM with $N=2400$", fontsize=14)
plt.xlim((-0.003, 0.063))
plt.legend()
plt.savefig("simulation_bistable_sbm.pdf")
