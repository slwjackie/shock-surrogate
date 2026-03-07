import numpy as np
import pandas as pd

df = pd.read_csv("data/meta.csv")

Utr_all = np.load("data/u_train.npz")["u"]
Upr_all = np.load("data/u_test_profile_ood.npz")["u"]
Umi_all = np.load("data/u_test_mismatch_ood.npz")["u"]

# split별 case_id는 0..N-1로 다시 시작하니까, split 내부에선 case_id를 index로 써도 OK
tr = df[df.split=="train"].iloc[0]
pr = df[df.split=="test_profile_ood"].iloc[0]
mi = df[df.split=="test_mismatch_ood"].iloc[0]

Utr = Utr_all[int(tr.case_id)]
Upr = Upr_all[int(pr.case_id)]
Umi = Umi_all[int(mi.case_id)]

print("Example params:")
print(" train dTdx,b,nu,k,E:", tr.dTdx, tr.b_quad, tr.nu, tr.k, tr.E)
print(" prof  dTdx,b,nu,k,E:", pr.dTdx, pr.b_quad, pr.nu, pr.k, pr.E)
print(" mis   dTdx,b,nu,k,E:", mi.dTdx, mi.b_quad, mi.nu, mi.k, mi.E)

print("L2 diff TRAIN vs PROFILE:", float(np.mean((Utr-Upr)**2)))
print("L2 diff TRAIN vs MISMATCH:", float(np.mean((Utr-Umi)**2)))
print("Ranges:")
print(" train min/max", float(Utr.min()), float(Utr.max()))
print(" prof  min/max", float(Upr.min()), float(Upr.max()))
print(" mis   min/max", float(Umi.min()), float(Umi.max()))