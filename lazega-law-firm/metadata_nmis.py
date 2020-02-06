import pickle
from utilities import nmi, ami, sorted_tuple, num_communities
import numpy as np

lines = open("ELattr.dat", "r").readlines()
seniority = []
status = []
gender = []
office = []
age = []
practice = []
law_school = []
for line in lines:
    # Group seniority and age metadata into 5-year bins to match Pamfil et al.
    x = tuple(int(v) for v in line.strip().split())
    status.append(x[1])
    gender.append(x[2])
    office.append(x[3])
    seniority.append(x[4] // 5)
    age.append(x[5] // 5)
    practice.append(x[6])
    law_school.append(x[7])

stable_memberships = pickle.load(open("lazega_stable_K_2-4.p", "rb"))
for i, memberships in enumerate(stable_memberships):
    part = memberships  # np.concatenate(memberships)
    print("=====Stable partition {} with K={}=====".format(i + 1, num_communities(part)))
    # print("Office: {:.3f}".format(nmi(part, office * 3)))
    # print("Practice: {:.3f}".format(nmi(part, practice * 3)))
    # print("Age: {:.3f}".format(nmi(part, age * 3)))
    # print("Seniority: {:.3f}".format(nmi(part, seniority * 3)))
    # print("Status: {:.3f}".format(nmi(part, status * 3)))
    # print("Gender: {:.3f}".format(nmi(part, gender * 3)))
    # print("Law School: {:.3f}".format(nmi(part, law_school * 3)))
    print("&{:.3f}".format(nmi(part, office * 3)))
    print("&{:.3f}".format(nmi(part, practice * 3)))
    print("&{:.3f}".format(nmi(part, age * 3)))
    print("&{:.3f}".format(nmi(part, seniority * 3)))
    print("&{:.3f}".format(nmi(part, status * 3)))
    print("&{:.3f}".format(nmi(part, gender * 3)))
    print("&{:.3f}".format(nmi(part, law_school * 3)))

    if i == 5:
        print(part)
