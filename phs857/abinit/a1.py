import numpy as np
import matplotlib.pyplot as plt

# ECUT TUNING
ecut_coarse, etotal_coarse = open("ecut_coarse.dat").read().split("\n\n")
ecut_fine, etotal_fine = open("ecut_fine.dat").read().split("\n\n")
ecut_coarse = np.array(list(map(float, ecut_coarse.split('\n'))))
etotal_coarse = np.array(list(map(float, etotal_coarse[:-1].split('\n'))))

ecut_fine = np.array(list(map(float, ecut_fine.split('\n'))))
etotal_fine = np.array(list(map(float, etotal_fine[:-1].split('\n'))))

ecut = np.concatenate([ecut_coarse, ecut_fine])
etotal_ecut = np.concatenate([etotal_coarse, etotal_fine])
ediff_ecut = etotal_ecut - etotal_ecut[-1]

idx = np.argsort(ecut)
ecut = ecut[idx]
etotal_ecut = etotal_ecut[idx]
ediff_ecut = ediff_ecut[idx]

# NGKPT TUNING
ngkpt = open("ngkpt.dat").read().split("\n")
ngkpt = np.array(list(map(float, ngkpt[:-1])))
diff_ngkpt = ngkpt - ngkpt[-1]

# PLOTTING
plt.scatter(ecut, etotal_ecut)
plt.xlabel("Energy Cutoff (Hartree)")
plt.ylabel("Total Energy (Hartree)")
plt.savefig('ecut.png')
plt.close()

plt.scatter(ecut, ediff_ecut)
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.xlabel("Energy Cutoff (Hartree)")
plt.ylabel("Energy Difference (Hartree)")
plt.savefig('ecut_diff.png')
plt.close()

plt.scatter(np.arange(len(ngkpt)), ngkpt)
plt.xlabel("Number of Grid Points for k Points Generation")
plt.ylabel("Energy (Hartree)")
plt.savefig('ngkpt.png')
plt.close()

plt.scatter(np.arange(ngkpt.shape[0]), diff_ngkpt)
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.xlabel("Number of Grid Points for k Points Generation")
plt.ylabel("Energy Difference (Hartree)")
plt.savefig('ngkpt_diff.png')
plt.close()
