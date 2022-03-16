import pandas as pd
from TopoPyScale import topoclass as tc
from TopoPyScale import topo_export as te
from TopoPyScale import topo_sim as sim
from TopoPyScale import topo_da as da
from TopoPyScale import topo_utils as ut
from datetime import datetime
import os
import numpy as np
from matplotlib import pyplot as plt

# you need to be in sim directory to run code
os.chdir("/home/joel/sim/tscale_projects/davos")

# temp keuyword to skip modis section
MODISprocess=False

startTime = datetime.now()
config_file = './config.yml'
mp = tc.Topoclass(config_file)
wdir = mp.config.project.directory



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

grid_stack = sim.topo_map_forcing(mp.downscaled_pts.t)
sim.write_ncdf(wdir, grid_stack, "T", "K", "air_temperature", mp.downscaled_pts.time)

mp.to_fsm()

# Simulate FSM
for i in range(mp.config.sampling.toposub.n_clusters):
    nsim = "{:0>2}".format(i)
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
ensemb_size = 100
startDA = "2018-03-01"
endDA = "2018-07-31"
perturb, perturb_uncor = da.ensemble_pars_gen(ensemb_size)
ensemb_type = "TP"

sdThresh = 5  # threshold for conversion to fSCA in cm

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

# extract fSCA timeseries
OBS = da.extract_fsca_timeseries(wdir, plot=True)

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
myobs = obs.fSCA / 100
R = 0.13

plt.plot(pred)
plt.plot(myobs)
plt.show()

W = da.PBS(myobs, pred, R)
plt.plot(W)
plt.show()

endTime2 = datetime.now()
da.da_plots(HX3D_weighted_mean, HX_HS3D_weighted_mean, W, mydates, myobs)

print("Runtime DA= " + (str(endTime2-startTime)))

# plot open loop against obs
plotDay = "2018-04-15"
df = sim.agg_by_var_fsm(5)
df_mean_open = sim.timeseries_means_period(df, plotDay, plotDay)
# plot weighted ensemble against obs
df2 = sim.agg_by_var_fsm_ensemble(5, W)
df_mean_da = sim.timeseries_means_period(df2, plotDay, plotDay)
import importlib
importlib.reload(da)
da.da_compare_plot(wdir, plotDay, df_mean_open, df_mean_da)