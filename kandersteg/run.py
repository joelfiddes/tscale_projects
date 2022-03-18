import pandas as pd
from TopoPyScale import topoclass as tc
from TopoPyScale import topo_export as te
from TopoPyScale import topo_sim as sim
from TopoPyScale import topo_da as da
from TopoPyScale import topo_utils as ut

import importlib
importlib.reload(da)

from datetime import datetime
import os
import numpy as np
from matplotlib import pyplot as plt

# you need to be in sim directory to run code
os.chdir("/home/joel/sim/tscale_projects/kandersteg")

# temp keuyword to skip modis section
MODISprocess=False

startTime = datetime.now()
config_file = './config.yml'
mp = tc.Topoclass(config_file)
mp.compute_dem_param()
mp.extract_topo_param()

# construct listpoints as convenience object
cluster_sizes = mp.toposub.kmeans_obj.counts_
listpoints = pd.DataFrame(mp.toposub.df_centroids)
listpoints["members"] = cluster_sizes
listpoints.to_csv("listpoints.csv")

mp.toposub.write_landform()
mp.compute_solar_geometry()
mp.compute_horizon()
mp.downscale_climate()
mp.to_fsm()

# Simulate FSM
for i in range(mp.config.sampling.toposub.n_clusters):
    nsim = "{:0>3}".format(i)
    sim.fsm_nlst(31, "./outputs/FSM_pt_"+ nsim +".txt", 24)
    sim.fsm_sim("./fsm_sims/nlst_FSM_pt_"+ nsim +".txt", "./FSM")

# extract GST results(7)
df = sim.agg_by_var_fsm(7)

# extraxt timeseries average
df_mean = sim.timeseries_means_period(df, mp.config.project.start, mp.config.project.end)

endTime = datetime.now()
print("Runtime = " + (str(endTime-startTime)))
# map to domain grid
sim.topo_map(df_mean)

# ensemble run (need to make a sim class to get self)


ensemble_dir = "./ensemble/"
if not os.path.exists(ensemble_dir):
    os.makedirs(ensemble_dir)

ds = mp.downscaled_pts
ensemb_size = 50
startDA = "2018-04-01"
endDA = "2018-07-31"
perturb = da.ensemble_pars_gen(ensemb_size)
ensemb_type = "TPSL"
wdir = mp.config.project.directory
sdThresh = 5 # threshold for conversion to fSCA in cm

# generate perturbation parameters
for N_ens in range(ensemb_size):
    ensembleN = "ENS" + str(N_ens) + "_"
    ds_perturb = da.ensemble_meteo_gen(ds, perturb, N_ens, ensemb_type)
    te.to_fsm(ds_perturb, fname_format= ensemble_dir + "/" +  ensembleN +  'FSM_pt_*.txt')

# Simulate FSM ensemble
for N_ens in range(ensemb_size):
    ensembleN = "ENS" + str(N_ens) + "_"
    for i in range(mp.config.sampling.toposub.n_clusters):
        nsim = "{:0>2}".format(i)
        sim.fsm_nlst(31, ensemble_dir + "/" +  ensembleN + "FSM_pt_"+ nsim +".txt", 24)
        sim.fsm_sim("./fsm_sims/nlst_" + ensembleN + "FSM_pt_" + nsim + ".txt", "./FSM")

# extract results convert to fSCA
HX, mydates = da.construct_HX(wdir, startDA, endDA)
HX[HX <= sdThresh] = 0
HX[HX > sdThresh] = 1
# convert HX to 3D array T, samples ensembles
HX3D = np.array(HX).reshape(HX.shape[0], ensemb_size, mp.config.sampling.toposub.n_clusters) #.transpose(0,2,1)
# weight ensemble by members
# step 1: multiply each sample by number of members and sum across Sample dimension
HX3D_weighted = (HX3D * np.array(listpoints.members)).sum(2)
# step 2: divide by total number of members/pixels to get weighted fSCA at domain llevel (0-1)
HX3D_weighted_mean = HX3D_weighted /np.sum(listpoints.members)

# extract results keep as SWE /HS
HX_HS, mydates = da.construct_HX(wdir, startDA, endDA)

# convert HX to 3D array T, samples ensembles
HX_HS3D = np.array(HX_HS).reshape(HX_HS.shape[0], ensemb_size, mp.config.sampling.toposub.n_clusters) #.transpose(0,2,1)
# weight ensemble by members
# step 1: multiply each sample by number of members and sum across Sample dimension
HX_HS3D_weighted_mean = (HX_HS3D * np.array(listpoints.members)).mean(2)



if MODISprocess:
    # get MODIS tile IDs
    latmax = mp.config.project.extent['latN']
    latmin = mp.config.project.extent['latS']
    lonmax = mp.config.project.extent['lonE']
    lonmin = mp.config.project.extent['lonW']

    vertmax, horizmax, vertmin, horizmin = da.modis_tile(latmax, latmin, lonmax, lonmin)

    # download MODIS (need to handle multiple tile case here with wrapper
    da.pymodis_download(wdir, int(vertmax), int(horizmin), startDA, endDA)

    # get domain parameters from landform
    epsg, bbox = da.projFromLandform(wdir + "/landform.tif")

    # process modis data
    da.process_modis(wdir, epsg, bbox)

# extrac fSCA timeseries
OBS = da.extract_sca_timeseries(wdir, plot=True)



# average across weighted clusters
#HX3D_weighted_mean = HX3D_weighted/ np.sum(listpoints.members)

plt.plot(HX3D_weighted_mean)
plt.show()

# check some results
metfile = "/home/joel/sim/topoPyscale_davos/outputs/FSM_pt_00.txt"
simfile = "/home/joel/sim/topoPyscale_davos/fsm_sims/sim_ENS0_FSM_pt_00.txt"

dfmet = ut.FsmMetParser(metfile)
dfsim = ut.FsmSnowParser(simfile)
ut.FsmPlot(dfmet)

# plot ensemble
ut.FsmPlot_ensemble('/home/joel/sim/topoPyscale_davos/', "HS", 0)

# Do PBS
obs = OBS[int(OBS.index[OBS['mydates'] == startDA].values) : int(OBS.index[OBS['mydates'] == endDA].values)+1]

# check we have identical timesteps
try:
    mydates.reset_index(drop=True,inplace=True)
    mydates.equals(obs.mydates)
    print("All " + str(len(mydates))+" timestamps are identical" )
except:
    print("Observation timesteps do not equal simulation timesteps, perhaps some days are missing from downloaded Satelitte timeseries")


# check set up by plotting HX and OBS

# must these be interval 0-1?!
pred = HX3D_weighted_mean
myobs = obs.fSCA/100
R = 0.13

plt.plot(pred)
plt.plot(myobs)
plt.show()

W = da.PBS(myobs, pred, R)

da.da_plots(HX3D_weighted_mean, HX_HS3D_weighted_mean, W, mydates, myobs)

# script from topo_utils (validation stuff)
from TopoPyScale import topo_da as da
map_path = "/home/joel/sim/topoPyscale_davos/landform.tif"
epsg_out, bbxox = da.projFromLandform(map_path)
sample = getCoordinatePixel(map_path, 9.80933, 46.82945, 4326, epsg_out)
filename = "/home/joel/sim/topoPyscale_davos/outputs/FSM_pt_" + str(sample-1)+ ".txt"
df = FsmMetParser(filename, freq="1h", resample=False)
wfj = SmetParser("/home/joel/data/wfj_optimal/WFJ_optimaldataset_v8.smet")

start_date = df.index[0]
end_date = df.index[-1]

#greater than the start date and smaller than the end date
mask = (wfj.index >= start_date) & (wfj.index <= end_date)
wfj = wfj.loc[mask]


def plot_xyline(ax):
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

ax = plt.subplot(2, 2, 1)
plt.scatter(df.TA, wfj.TA, alpha=0.3)
#r, bias, rmse = mystats(df.TA, wfj.TA)
#plt.title("uncorrected rmse=" + str(round(rmse, 2)))
plt.xlabel("Modelled")
plt.ylabel("Measured")
plot_xyline(ax)


ax = plt.subplot(2, 2, 2)
plt.scatter(df.ISWR, wfj.ISWR, alpha=0.3)
#r, bias, rmse = mystats(df.TA, wfj.TA)
#plt.title("uncorrected rmse=" + str(round(rmse, 2)))
plt.xlabel("Modelled")
plt.ylabel("Measured")
plot_xyline(ax)


ax = plt.subplot(2, 2, 3)
plt.scatter(df.RH, wfj.RH*100, alpha=0.3)
#r, bias, rmse = mystats(df.TA, wfj.TA)
#plt.title("uncorrected rmse=" + str(round(rmse, 2)))
plt.xlabel("Modelled")
plt.ylabel("Measured")
plot_xyline(ax)

ax = plt.subplot(2, 2, 4)
plt.scatter(df.ILWR, wfj.ILWR, alpha=0.3)
#r, bias, rmse = mystats(df.TA, wfj.TA)
#plt.title("uncorrected rmse=" + str(round(rmse, 2)))
plt.xlabel("Modelled")
plt.ylabel("Measured")
plot_xyline(ax)
plt.show()

myfile = "/home/joel/sim/topoPyscale_davos/outputs/FSM_pt_0.txt"
df = FsmMetParser(myfile)
FsmPlot(df)

myfile = "/home/joel/sim/topoPyscale_davos/fsm_sims/sim_ENS1_FSM_pt_0.txt"
df = FsmSnowParser(myfile)
FsmPlot(df)
