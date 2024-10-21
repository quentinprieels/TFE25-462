from radcomlib.radar_toolbox import compute_CFAR_threshold

from numpy import arange,zeros

from matplotlib.pyplot import figure,plot,xlabel,ylabel

P_FA_vec = [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]
kind = "CA"
order = 0.75
N_vec = arange(10,1001,1)

thresh = zeros((len(N_vec),len(P_FA_vec)))

for j,P_FA in enumerate(P_FA_vec):
    for i,N in enumerate(N_vec):
        thresh[i,j] = compute_CFAR_threshold(P_FA,N,kind=kind,order=order,save=True,verbose=True)
    
figure()
for j in range(len(P_FA_vec)):
    plot(N_vec,thresh)
xlabel("N")
ylabel("CFAR coefficient")