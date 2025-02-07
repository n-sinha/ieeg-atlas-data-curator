# %% [markdown]
# # Book to evaluate Weighted Standard Distance

# %% [markdown]
# ## imports

# %%
import os
import sys
import re

code_path = os.path.dirname(os.getcwd())
sys.path.append(code_path)

import warnings
import json
from os.path import join as ospj

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2
from scipy.io import loadmat
from tqdm import tqdm

import tools

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools

from sklearn import *
import statsmodels.api as sm
from scipy.stats import pearsonr
from confidenceinterval import roc_auc_score

# os.environ['PYTHONWARNINGS']='ignore' # for parallel

# warnings.filterwarnings("ignore")
sns.set_theme(
    context="notebook",
    palette="deep",
    style="white",
    rc={
        "axes.spines.right": True,
        "axes.spines.top": True,
        "xtick.bottom": True,
        "ytick.left": True,
    },
)

# %% [markdown]
# ### set Params

# %%
with open("../config.json", "rb") as f:
    config = json.load(f)
repo_path = config["repositoryPath"]
metadata_path = ospj(repo_path, "ieeg-metadata")
data_path = ospj(repo_path, "data")
figpath = config["figure_path"]

# %%
PreImplantData = pd.read_excel(
    ospj(metadata_path, "Preimplant data-final.xlsx"), index_col=2
)

dropcols = [
    "Done?",
    "annotator",
    "Notes",
    "mri_lesion_ifother",
    "spect_lesion_yn",
    "SOZ electrode",
    "SOZ localization",
    "confidence",
    "who_clinician",
    "Notes_clinician",
    "region_clinician",
    "lateralization_clinician",
]
PreImplantData.drop(columns=dropcols, inplace=True)
PreImplantData = PreImplantData.applymap(
    lambda s: s.lower().strip() if type(s) == str else s
)
PreImplantData.dropna(how="all", inplace=True)
# for these cols, NA means no spikes; for all other cols NA means test not performed
PreImplantData[["eeg_inter_lat", "eeg_inter_loc", "eeg_ictal_lat", "eeg_ictal_loc"]] = (
    PreImplantData[
        ["eeg_inter_lat", "eeg_inter_loc", "eeg_ictal_lat", "eeg_ictal_loc"]
    ].fillna("none")
)


PreImplantData = PreImplantData.astype("category")

# define target:
target_col = "postimplant_soz"
PreImplantData = PreImplantData.drop(
    PreImplantData[PreImplantData[target_col].isna()].index
)
Y = (PreImplantData[target_col] == "unifocal").astype(int).rename("focality")  # binary
# Y = PreImplantData[target_col].rename('focality') #categorical, multiclass
Y

# %%
outcomes = pd.read_excel(
    ospj(
        data_path,
        "CNTSurgicalRepositor-Erinbasicdemographic_DATA_LABELS_2022-10-31_1508.xlsx",
    )
)
outcomes.index = (
    outcomes["HUP Number"].map(lambda x: "HUP" + str(x).zfill(3)).rename("name")
)

# %%
all_procs = pd.get_dummies(outcomes["Type of Surgery:"].str.split(",").explode())
all_procs = all_procs.groupby(all_procs.index).sum().rename_axis("surgery", axis=1)
Surgery = (
    all_procs["Laser ablation"]
    | all_procs["Resection with intracranial implant"]
    | all_procs["Resection without intracranial implant"]
)
Device = all_procs["VNS"] | all_procs["DBS"] | all_procs["RNS"]
Other = all_procs["Other"]
type_of_proc = pd.concat(
    [Surgery, Device, Other], axis=1, keys=["surgery", "device", "other"]
)

# %%
# what procedure did they get
type_of_proc = type_of_proc.apply(
    lambda x: (
        "both"
        if ((x["surgery"]) & (x["device"]))
        else "surgery" if x["surgery"] else "device" if x["device"] else "none"
    ),
    axis=1,
)

# %%
bands = [
    [0.5, 4],  # delta
    [4, 8],  # theta
    [8, 12],  # alpha
    [12, 30],  # beta
    [30, 80],  # gamma
    [0.5, 80],  # broad
]
band_names = ["delta", "theta", "alpha", "beta", "gamma", "broad"]
N_BANDS = len(bands)

# %% [markdown]
# ## Replicate 5sense

# %%
FiveSensePreds = [
    "mri_lesion_yn",
    "mri_lesion_lat",
    "mri_lesion_loc",
    "eeg_inter_lat",
    "eeg_inter_loc",
    "neuropsy_lat",
    "semiology_loc",
    "eeg_ictal_lat",
    "eeg_ictal_loc",
]
X = PreImplantData[FiveSensePreds]

# %%
# define predictors:
FiveSensePreds = [
    "mri_lesion_yn",
    "mri_lesion_lat",
    "mri_lesion_loc",
    "eeg_inter_lat",
    "eeg_inter_loc",
    "neuropsy_lat",
    "semiology_loc",
    "eeg_ictal_lat",
    "eeg_ictal_loc",
]
X = PreImplantData[FiveSensePreds]
# reformat data into their schema
MRIdict = {
    "condition": [
        X["mri_lesion_yn"] == "n",
        X["mri_lesion_lat"].isin(["left", "right", "other"])
        & X["mri_lesion_loc"].isin(["temporal", "frontal", "other"]),
    ],
    "choice": ["nolesion", "focal"],
    "default": "nonfocal",
}


interDict = {
    "condition": [
        (X["eeg_inter_lat"] == "none") & (X["eeg_inter_loc"] == "none"),
        X["eeg_inter_lat"] == "bilateral",
    ],
    "choice": ["nospike", "bilateral"],
    "default": "allothers",
}
SemioDict = {
    "condition": [X["semiology_loc"].isin(["temporal", "frontal", "other"])],
    "choice": ["localizing"],
    "default": "nonlocalizing",
}
NeuroPsyDict = {
    "condition": [
        X["neuropsy_lat"].isin(["left", "right"]),
        X["neuropsy_lat"].isin(["bilateral", "multifocal"]),
    ],
    "choice": ["localizing", "nonlocalizing"],
    "default": "no_deficit",
}
ictalDict = {
    "condition": [
        (X["eeg_ictal_lat"].isin(["left", "right"]))
        & (X["eeg_ictal_loc"].isin(["temporal", "frontal", "other"])),
        (X["eeg_ictal_lat"] == "bilateral")
        & (~X["eeg_ictal_loc"].isin(["unclear", "broad", "multifocal"])),
    ],
    "choice": ["focal", "bilateral"],
    "default": "none",
}


X["MRI"] = np.select(
    MRIdict["condition"], MRIdict["choice"], default=MRIdict["default"]
)
X["InterEEG"] = np.select(
    interDict["condition"], interDict["choice"], default=interDict["default"]
)
X["Semio"] = np.select(
    SemioDict["condition"], SemioDict["choice"], default=SemioDict["default"]
)
X["Ictal"] = np.select(
    ictalDict["condition"], ictalDict["choice"], default=ictalDict["default"]
)
X["NeuroPsy"] = np.select(
    NeuroPsyDict["condition"], NeuroPsyDict["choice"], default=NeuroPsyDict["default"]
)
FiveSenseDf = pd.get_dummies(X[["MRI", "InterEEG", "Semio", "Ictal", "NeuroPsy"]])
FiveSenseDf

# %%
weights = np.array(
    [
        0,
        -2.2636,
        -2.1494,
        1.1807,
        0,
        1.8056,
        0.8489,
        0,
        0,
        0.8442,
        -0.8124,
        0,
        1.15,
        -0.26,
    ]
)


for i in range(len(weights)):
    print(FiveSenseDf.columns[i], " : ", weights[i])

# %%
score = np.matmul(FiveSenseDf.to_numpy(), weights) - 0.3135

# %%
from sklearn import metrics

# %%
prediction = np.exp(score) / (np.exp(score) + 1)
yhat = prediction > 0.376
fpr, tpr, threshold = metrics.roc_curve(Y, prediction)
plt.plot(fpr, tpr)
plt.xlabel("fpr")
plt.ylabel("tpr")

# %%
metrics.roc_auc_score(Y, prediction)

# %%
print(metrics.classification_report(Y, yhat))

# %%
cfmx = pd.crosstab(Y, yhat)
cfmx

# %%
metrics.balanced_accuracy_score(Y, yhat)

# %%
FiveSenseScore = (
    pd.DataFrame(score, index=Y.index.rename("patient"), columns=["fivesense"])
    .join(Y)
    .set_index("focality", append=True)
)
FiveSenseScore.to_csv(ospj(data_path, "5_Sense_Score.csv"))

# %%
FiveSensePreds = (
    pd.DataFrame(prediction, index=Y.index.rename("patient"), columns=["fivesense"])
    .join(Y)
    .set_index("focality", append=True)
)

# %%
plt.figure(figsize=(2, 5))
sns.boxplot(
    data=FiveSenseScore.reset_index(), x="focality", y="fivesense", palette="pastel"
)
sns.swarmplot(
    data=FiveSenseScore.reset_index(),
    x="focality",
    y="fivesense",
    hue="focality",
    palette="dark",
    legend=False,
)
plt.xticks(ticks=[0, 1], labels=["nonfocal", "focal"])
plt.xlabel("")

# %%
mannwhitneyu(*[group for i, group in FiveSenseScore.groupby("focality")])

# %% [markdown]
# ## Compile Data

# %% [markdown]
# ### Get Data
# - Norm Atlas
# - Patient data
#     - bipolar
#         - bandpower
#             - raw to compare
#             - canonical for actual tests
#     - electrode locs
# 

# %%
norm_logpower = pd.read_pickle(ospj(data_path, "norm_logpower.pkl"))

# %%
norm_power = pd.read_pickle(ospj(data_path, "Norm_Power.pkl"))
norm_power

# %%
AllCoords = pd.read_pickle(ospj(data_path, "AllCoords.pkl"))
DropLabels = [
    "Unknown",
    "Left-Cerebral-White-Matter",
    "Left-Lateral-Ventricle",
    "Left-Inf-Lat-Vent",
    "Left-Cerebellum-White-Matter",
    "Left-Cerebellum-Cortex",
    "3rd-Ventricle",
    "4th-Ventricle",
    "Brain-Stem",
    "CSF",
    "Left-choroid-plexus",
    "Right-Cerebral-White-Matter",
    "Right-Lateral-Ventricle",
    "Right-Inf-Lat-Vent",
    "Right-Cerebellum-White-Matter",
    "Right-Cerebellum-Cortex",
    "Right-choroid-plexus",
    "WM-hypointensities",
    "Optic-Chiasm",
    "CC_Posterior",
    "CC_Mid_Posterior",
    "CC_Central",
    "CC_Mid_Anterior",
    "CC_Anterior",
    "ctx-lh-unknown",
    "ctx-lh-corpuscallosum",
    "ctx-rh-unknown",
    "ctx-rh-corpuscallosum",
    "Left-VentralDC",
    "Left-vessel",
    "Right-VentralDC",
    "Right-vessel",
]
LUT = pd.read_csv(
    ospj(metadata_path, "roiDKT.csv"),
).rename(columns={"Lobe": "lobe"})
LUT["roi"] = LUT["Abbvr"].map(lambda x: x[: int(x.find("_"))])
LUT["lat"] = LUT["isSideLeft"].map({1: "left", 0: "right"})

# %%
AllCoords

# %%
ChansToGet = AllCoords.dropna(subset="label")
ChansToGet = ChansToGet[
    (~ChansToGet["label"].isin(DropLabels))
    & (ChansToGet["label"].isin(norm_power["region"]))
    & (ChansToGet["patient"].isin(Y.index))
]
ChansToGet

# %%
AllPtPxx = pd.read_pickle(ospj(data_path, "AllPtPxxBipolar.pkl"))
AllPtPxx = AllPtPxx  # .xs()
AllPtPxx

# %%
NormNxx = pd.read_pickle(ospj(data_path, "NormAllConnect.pkl"))

# %%
NormNxx = NormNxx[NormNxx.reset_index("num")["num"] > 8]
NormNxx = NormNxx[~(NormNxx["std"] == 0).any(axis=1)]
NormNxx = NormNxx.stack("band").reset_index("num")
# symmetric indices for all connections
AllNormConns = list(
    NormNxx.reset_index()[["roi_from", "roi_to"]].itertuples(index=False, name=None)
)

# %%
AllPtNxx = pd.read_pickle(ospj(data_path, "AllPtNxxBipolar.pkl")).reset_index()

# %%
# find combos where both channel 1 and channel 2 are in ROIs we care about
AllConnCombos = pd.merge(
    AllPtNxx[["patient", "channel_1", "channel_2"]].drop_duplicates(),
    ChansToGet[["patient", "name", "label"]].drop_duplicates(),
    left_on=["patient", "channel_1"],
    right_on=["patient", "name"],
).drop("name", axis=1)
# did it in two merges to use the suffixes
AllConnCombos = pd.merge(
    AllConnCombos,
    ChansToGet[["patient", "name", "label"]].drop_duplicates(),
    left_on=["patient", "channel_2"],
    right_on=["patient", "name"],
    suffixes=("_1", "_2"),
).drop("name", axis=1)
# filter patient channel combinations for only those in Normative atlas
AllConnCombos = AllConnCombos[
    [
        (conn in AllNormConns)
        for conn in list(zip(AllConnCombos.label_1, AllConnCombos.label_2))
    ]
]
# merge all data to get roi labels
AllPtNxx = AllPtNxx.merge(AllConnCombos)

# %%
mannwhitneyu(
    *[
        group
        for i, group in AllPtNxx.groupby("patient")
        .apply(lambda x: len(np.unique(x[["channel_1", "channel_2"]].values.ravel())))
        .to_frame()
        .join(Y)
        .groupby("focality")
    ]
)

# %% [markdown]
# ### filter channels

# %%
AllChans = pd.merge(
    AllPtPxx.reset_index(),
    ChansToGet,
    left_on=["patient", "channel"],
    right_on=["patient", "name"],
)

# %%
AllChans = AllChans.merge(Y, left_on="patient", right_index=True)

# %%
AllChans[["patient", "channel", "soz"]].drop_duplicates()["soz"].value_counts()

# %%
all_power_bands = AllChans.groupby(
    ["patient", "clip", "channel", "period", "focality"], group_keys=True
)[["freq", "bandpower"]].apply(lambda x: tools.format_bandpower(x))
all_power_bands
all_power_bands.to_pickle(ospj(data_path, "PxxByBandBipolar.pkl"))
# all_power_bands = pd.read_pickle(ospj(data_path,'PxxByBandBipolar.pkl'))

# %%
all_power_bands

# %%
def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    return (u1 - u2) / s

# %% [markdown]
# ## Main

# %% [markdown]
# ### predictors of focality

# %% [markdown]
# #### Zscore bandpower to norm atlas by ROI

# %%
Mean = norm_power.groupby(["region", "band"])["bandpower"].mean().rename("mean")
Std = norm_power.groupby(["region", "band"])["bandpower"].std().rename("std")
NormRois = pd.concat([Mean, Std], axis=1)

# %%
def getZscore(x, l, b):
    return (x - NormRois.loc[l, b]["mean"]) / NormRois.loc[l, b]["std"]

# %%
AllBandChans = pd.merge(
    all_power_bands.stack("band").rename("bandpower").reset_index(),
    ChansToGet,
    left_on=["patient", "channel"],
    right_on=["patient", "name"],
)
AllBandChans["focality"] = AllBandChans["patient"].map(lambda x: Y[x])
AllBandChans

# %%
# get Z scores of bandpower and drop nas
AllBandChans["ZScore"] = AllBandChans.groupby(
    ["label", "band"], as_index=False, group_keys=False
).apply(lambda x: getZscore(x["bandpower"], x.name[0], x.name[1]))
AllBandChans.dropna(subset="ZScore", inplace=True)
AllBandChans["absZScore"] = AllBandChans["ZScore"].abs()
# set multiindex to identifiers of patient/channel/period/clip; only values are bandpower, Z, abs(Z)
AllBandChans = (
    AllBandChans.set_index(
        [
            col
            for col in AllBandChans.columns
            if col not in ["ZScore", "absZScore", "bandpower"]
        ]
    )
    .swaplevel("band", "soz")
    .droplevel(
        [
            "name",
            "x",
            "y",
            "z",
            "index",
        ]
    )
)

# %%
AllBandChans[["ZScore", "absZScore"]].xs("interictal", level="period").groupby(
    ["patient", "focality", "band", "channel", "soz"]
).median().reset_index().groupby(["patient", "channel"]).nunique()

# %% [markdown]
# #### Standard Distance

# %%
def StandDist(X):
    """
    take m x 3 df (x, y, z) and outputs standard distance
    """
    return np.sqrt(X.var(ddof=0).sum())


def WeightStandDist(w, X):
    """
    takes m x 1 weights and m x 3 df (x, y, z) and outputs weighted standard distance
    """
    Xbar = X.multiply(w, axis=0).sum().divide(w.sum())  # mean center
    return np.sqrt(
        (X.subtract(Xbar) ** 2).multiply(w, axis=0).sum().divide(w.sum()).sum()
    )

# %%
Base_SD = (
    AllBandChans.reset_index()[
        ["patient", "focality", "channel", "x_mm", "y_mm", "z_mm"]
    ]
    .drop_duplicates()
    .groupby(["patient", "focality"])[["x_mm", "y_mm", "z_mm"]]
    .apply(lambda x: StandDist(x))
    .rename("Unweighted")
)

# %% [markdown]
# ##### Weighted standard distance

# %%
BP_WSD_all_clips = (
    AllBandChans.xs("interictal", level="period")
    .reset_index()
    .groupby(["patient", "band", "focality", "clip"])
    .apply(
        lambda x: (WeightStandDist(x["absZScore"], x[["x_mm", "y_mm", "z_mm"]]))
    )  # .mask(lambda x: x>2,0)
)
# BP_WSD = (
#     BP_WSD.groupby([col for col in BP_WSD.index.names if col != "clip"])
#     .describe()
#     .drop("count", axis=1)
#     .unstack('band')
# )
BP_WSD = (
    BP_WSD_all_clips.groupby(["patient", "band", "focality"]).median().unstack("band")
)

# %%
BP_WSD[band_names].join(Base_SD).melt(ignore_index=False).reset_index()

# %% [markdown]
# #### Z score coherence

# %%
def getZscore(x, r1, r2, b):
    return (x - NormNxx.loc[r1, r2, b]["mean"]) / NormNxx.loc[r1, r2, b]["std"]

# %%
AllPtNxx = AllPtNxx[~AllPtNxx["coherence"].isna()]

AllPtNxx["ZScore"] = AllPtNxx.groupby(
    ["label_1", "label_2", "band"], as_index=False, group_keys=False
).apply(lambda x: getZscore(x["coherence"], x.name[0], x.name[1], x.name[2]))

AllPtNxx[["roi_1", "roi_2"]] = np.sort(AllPtNxx[["label_1", "label_2"]].values)
AllPtNxx.drop(["label_1", "label_2"], axis=1, inplace=True)

AllPtNxx = AllPtNxx.merge(Y, left_on="patient", right_index=True)
AllPtNxx["absZScore"] = AllPtNxx["ZScore"].abs()

# %%
## duplicates for some reason, possibly channel name overlap
AllPtNxx = AllPtNxx.drop(
    AllPtNxx[
        AllPtNxx[
            ["patient", "channel_1", "channel_2", "period", "band", "clip"]
        ].duplicated()
    ].index
)

# %% [markdown]
# ##### getting channel-wise stats then aggregating over clips

# %%
def GetChannelStats(df):
    chans = np.unique(df[["channel_1", "channel_2"]].values.ravel())
    newdf = []
    for chan in chans:
        newdf.append(
            df[(df["channel_1"] == chan) | (df["channel_2"] == chan)][
                ["absZScore"]
            ].quantile(0.75)
        )
    return pd.concat(newdf, names=["channel"], keys=chans)


### within clips describe the distributions per channel, then aggregate
## with or without band?
# within each clip and channel
Nxx_Chan_Stats = AllPtNxx[AllPtNxx["period"] == "interictal"]
Nxx_Chan_Stats = Nxx_Chan_Stats.groupby(["patient", "focality", "band", "clip"]).apply(
    lambda x: GetChannelStats(x)
)

# %%
Nxx_Chan_Stats

# %% [markdown]
# ### all predictors

# %%
Coh_Mean_ZScore = (
    Nxx_Chan_Stats.unstack()
    .merge(
        ChansToGet.set_index(["patient", "name"]),
        left_on=["patient", "channel"],
        right_index=True,
    )
    .groupby(["patient", "focality", "band", "clip"])["absZScore"]
    .apply(lambda x: pd.Series({"mean": x.mean(), "std": x.std()}))
    .groupby(["patient", "focality", "band"])
    .median()
    .unstack("band")
)
Coh_Mean_ZScore.columns = [col + "_mean_coh" for col in Coh_Mean_ZScore.columns]

# %%
## Then get weighted standard distance
Coh_WSD_all_clips = (
    Nxx_Chan_Stats.unstack()
    .merge(
        ChansToGet.set_index(["patient", "name"])[["x_mm", "y_mm", "z_mm"]],
        left_on=["patient", "channel"],
        right_index=True,
    )
    .groupby(["patient", "focality", "band", "clip"])
    .apply(lambda x: WeightStandDist(x["absZScore"], x[["x_mm", "y_mm", "z_mm"]]))
)
Coh_WSD = (
    Coh_WSD_all_clips.groupby(["patient", "focality", "band"])
    .median()
    .rename("Adj_WSD")
    .unstack("band")
)

# g = pd.read_pickle(ospj(metadata_path, 'Nxx-ChannConn_Brain_Clips_WSD'))

# %% [markdown]
# ### ML Model

# %%
def ML_Model(X, y):
    """X is a pd.DataFrame and y is a same indexed pd.Series"""
    n_folds = 101
    n_repeats = 1
    scaler = preprocessing.StandardScaler()

    # classifier = svm.SVC(kernel='linear', C=10, random_state=42, probability=True)
    classifier = linear_model.LogisticRegression(
        solver="liblinear", penalty="l1", random_state=42
    )

    pipe = pipeline.Pipeline([("scale", scaler), ("clf", classifier)])

    kfcv = model_selection.StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=42
    )
    loo = model_selection.LeaveOneOut()
    rkfcv = model_selection.RepeatedStratifiedKFold(
        n_splits=n_folds, n_repeats=n_repeats, random_state=43
    )

    YTEST = []
    YHAT = []
    YPRED = []
    FPR = []
    TPR = []
    THRESH = []
    coefs = []
    i = 0
    for train_index, test_index in loo.split(X, y):
        i += 1
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if X.ndim == 1:
            X_train = X_train.to_numpy().reshape(-1, 1)
            X_test = X_test.to_numpy().reshape(-1, 1)

        # clf = model_selection.GridSearchCV(estimator = pipe, param_grid = param_grid, scoring = 'roc_auc', cv = kfcv, n_jobs=).fit(X_train,y_train)
        # clf = feature_selection.RFECV(estimator=pipe, cv = model_selection.StratifiedKFold(n_splits=10, shuffle=True), scoring='balanced_accuracy', importance_getter="named_steps.clf.coef_")
        clf = pipe.fit(X_train, y_train)
        y_hat = clf.predict(X_test)
        y_pred = clf.predict_proba(X_test)[:, 1]
        # y_pred = clf.decision_function(X_test)
        coefs.append(clf.named_steps.clf.coef_)
        YTEST.append(y_test)
        YHAT.append(y_hat)
        YPRED.append(y_pred)

        if i % n_folds == 0:
            y_true_fold = pd.concat(YTEST[-n_folds:])
            y_pred_fold = np.concatenate(YPRED[-n_folds:])
            fpr, tpr, thresh = metrics.roc_curve(y_true_fold, y_pred_fold)
            FPR.append(fpr)
            TPR.append(tpr)
    YHAT = np.concatenate(YHAT)
    YPRED = np.concatenate(YPRED)
    YTEST = pd.concat(YTEST)

    AvgPreds = (
        YTEST.to_frame()
        .join(pd.Series(YPRED, index=YTEST.index).rename("prediction"))
        .groupby(["patient", "focality"])
        .mean()
        .reset_index("focality")
    )

    roc = metrics.roc_auc_score(AvgPreds["focality"], AvgPreds["prediction"])
    bac = metrics.balanced_accuracy_score(YTEST, YHAT)

    interp_tprs = []
    for f, t in zip(FPR, TPR):
        fpr_mean = np.linspace(0, 1, 100)
        interp_tpr = np.interp(fpr_mean, f, t)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)
    ROC = pd.DataFrame(np.transpose(interp_tprs), index=fpr_mean)

    return (
        {"roc_score": roc, "bac": bac},
        ROC,
        {"true": YTEST, "pred": YPRED, "yhat": YHAT},
        pd.DataFrame(np.array(coefs).squeeze(), columns=clf.feature_names_in_),
    )

# %%
def PrettyPictures(exog, endog, name=None, scale=False):
    if not name:
        name = endog.name
    if scale:
        exog = (
            preprocessing.MinMaxScaler()
            .set_output(transform="pandas")
            .fit_transform(exog)
        )
    data = exog.melt(ignore_index=False, var_name="variable").join(
        endog.rename(name), how="inner"
    )
    plt.figure()
    ax = sns.boxplot(
        data=data,
        x="variable",
        y="value",
        hue=name,
        palette="pastel",
        showfliers=False,
    )
    sns.stripplot(
        data=data,
        x="variable",
        y="value",
        hue=name,
        palette="dark",
        alpha=0.8,
        dodge=True,
        legend=False,
        ax=ax,
    )

    plt.xticks(rotation=30)
    plt.ylabel("probability focal")

    df2 = (
        data.groupby("variable")
        .apply(
            lambda Test: pd.Series(
                {
                    "cohend": cohend(
                        *[group.value for (j, group) in Test.groupby(name)]
                    ),
                    "u-stat": mannwhitneyu(
                        *[group.value for (j, group) in Test.groupby(name)]
                    )[0],
                    "ranksum": mannwhitneyu(
                        *[group.value for (j, group) in Test.groupby(name)]
                    )[1],
                    "roc": real_roc(Test[name], Test["value"])[0],
                    "ci": real_roc(Test[name], Test["value"])[1],
                }
            )
        )
        .sort_values("ranksum", ascending=True, key=abs)
    )
    display(df2)
    return (
        df2.melt(var_name="metric", value_name=name, ignore_index=False).set_index(
            "metric", append=True
        ),
        ax,
    )

# %%
## take the confidence interval returned by roc_auc_score and use 1-ci if the midpoint is less than 0.5
def real_roc(y_true, y_score):
    roc = metrics.roc_auc_score(y_true, y_score)
    ci = roc_auc_score(y_true, y_score, ci_method="roc_auc")[1]
    if roc < 0.5:
        roc = 1 - roc
        ci = (1 - ci[1], 1 - ci[0])
    return (roc, ci)

# %%
def MakePredictions(exog, endog, fname=None):
    exog = (
        preprocessing.StandardScaler()
        .set_output(transform="pandas")
        .fit_transform(exog)
    )
    X_reduced_1 = exog["fivesense"]
    X_reduced_2 = exog[["fivesense", "Unweighted"]]

    # exog = (preprocessing.StandardScaler()
    #         .set_output(transform="pandas")
    #         .fit_transform(exog))

    res = sm.Logit(
        endog,
        sm.add_constant(exog),
    ).fit()
    # ).fit_regularized(alpha = 0.99)
    preds = res.predict()
    print(res.summary())
    if fname:
        with open(ospj(figpath, f"{fname}.csv"), "w") as fh:
            fh.write(res.summary().as_csv())

    full_ll = res.llf
    full_df = res.df_model

    reduced_res_1 = sm.Logit(endog, X_reduced_1).fit()
    reduced_ll = reduced_res_1.llf
    reduced_df = reduced_res_1.df_model

    LR_stat = -2 * (reduced_ll - full_ll)
    print("FiveSense:")
    print(f"LR: {LR_stat:0.2f}, chi2: {chi2.sf(LR_stat, full_df - reduced_df):0.4f}")

    reduced_res_2 = sm.Logit(endog, sm.add_constant(X_reduced_2)).fit()
    reduced_ll = reduced_res_2.llf
    reduced_df = reduced_res_2.df_model
    # preds_unweighted = reduced_res_2.predict()

    LR_stat = -2 * (reduced_ll - full_ll)
    print("FiveSense and Unweighted")
    print(f"LR: {LR_stat:0.2f}, chi2: {chi2.sf(LR_stat,full_df - reduced_df):0.4f}")
    print(f"all: {res.aic}, 5s: {reduced_res_1.aic}, baseline: {reduced_res_2.aic}")

    return preds

# %%
def PlotLogit(X, y, name=None):
    if not name:
        name = y.name
    preds = X["IEEG Model Prediction"]
    data = X.join(y)

    plt.figure()
    f = sns.JointGrid(data=data, x="fivesense", y="IEEG Model Prediction", hue=name)
    f.figure.tight_layout()
    f.figure.subplots_adjust(top=0.92)
    f.figure.suptitle(
        "IEEG Model vs. Five Sense, r = {:0.2f}".format(
            np.corrcoef(preds, X["fivesense"])[0, 1]
        )
    )
    f.plot_joint(sns.scatterplot)
    sns.regplot(
        data,
        x=f.x,
        y=f.y,
        scatter=False,
        ax=f.ax_joint,
        color="tab:gray",
        scatter_kws={"palette": "muted"},
    )
    sns.boxplot(
        data,
        x=f.hue,
        y=f.y,
        ax=f.ax_marg_y,
        orient="v",
        palette="pastel",
        showfliers=False,
    )
    sns.swarmplot(
        data, x=f.hue, y=f.y, ax=f.ax_marg_y, orient="v", palette="deep", alpha=0.8
    )
    sns.boxplot(
        data,
        y=f.hue,
        x=f.x,
        ax=f.ax_marg_x,
        orient="h",
        palette="pastel",
        showfliers=False,
    )
    sns.swarmplot(
        data, y=f.hue, x=f.x, ax=f.ax_marg_x, orient="h", palette="deep", alpha=0.8
    )
    f.set_axis_labels(xlabel="5-Sense Score", ylabel="IEEG Model")

    plt.figure()
    g = sns.JointGrid(
        data=data,
        x="Preimplant Baseline Prediction",
        y="IEEG Model Prediction",
        hue=name,
    )
    g.figure.tight_layout()
    g.figure.subplots_adjust(top=0.92)
    g.figure.suptitle(
        "IEEG Model vs. Implant Baseline, r = {:0.2f}".format(
            np.corrcoef(preds, X["Preimplant Baseline Prediction"])[0, 1]
        )
    )
    g.set_axis_labels(xlabel="Implant Baseline", ylabel="IEEG Model")
    g.plot_joint(sns.scatterplot)
    sns.regplot(
        data,
        x=g.x,
        y=g.y,
        scatter=False,
        ax=g.ax_joint,
        color="tab:gray",
        scatter_kws={"palette": "muted"},
    )
    sns.boxplot(
        data,
        x=g.hue,
        y=g.y,
        ax=g.ax_marg_y,
        orient="v",
        palette="pastel",
        showfliers=False,
    )
    sns.swarmplot(
        data, x=g.hue, y=g.y, ax=g.ax_marg_y, orient="v", palette="deep", alpha=0.8
    )
    sns.boxplot(
        data,
        y=g.hue,
        x=g.x,
        ax=g.ax_marg_x,
        orient="h",
        palette="pastel",
        showfliers=False,
    )
    sns.swarmplot(
        data, y=g.hue, x=g.x, ax=g.ax_marg_x, orient="h", palette="deep", alpha=0.8
    )

    return f, g
    # plt.figure()
    # g = sns.lmplot(data, x = 'gamma_bp', y = 'all_data', hue = name, fit_reg= False)
    # sns.regplot(data, x = 'gamma_bp', y = 'all_data', scatter=False, ax=g.axes[0, 0], color = 'g')
    # plt.ylim(0,1.03)

    # plt.ylabel('post test proba')
    # plt.title('pre vs. post-test proba, r = {:0.2f}'.format(np.corrcoef(preds, X['gamma_bp'])[0,1]))
    # plt.show()

# %% [markdown]
# #### Out of Box

# %%
All_X = (
    BP_WSD.join(Coh_WSD, how="inner", lsuffix="_bp", rsuffix="_coh")
    .join(Base_SD, how="inner")
    .join(FiveSenseScore, how="inner")  # pre-logit score
)

# %%
All_Patient = All_X.index

# %%
def PlotDecision(df, x, y, hue):
    plt.figure()
    g = sns.JointGrid(data=df, x=x, y=y, hue=hue)
    sns.regplot(
        All_X,
        x=g.x,
        y=g.y,
        scatter=False,
        ax=g.ax_joint,
        color="tab:gray",
        scatter_kws={"palette": "muted"},
    )
    sns.boxplot(
        All_X,
        x=g.hue,
        y=g.y,
        ax=g.ax_marg_y,
        orient="v",
        palette="pastel",
        showfliers=False,
    )
    sns.swarmplot(
        All_X, x=g.hue, y=g.y, ax=g.ax_marg_y, orient="v", palette="deep", alpha=0.8
    )
    sns.boxplot(
        All_X,
        y=g.hue,
        x=g.x,
        ax=g.ax_marg_x,
        orient="h",
        palette="pastel",
        showfliers=False,
    )
    sns.swarmplot(
        All_X, y=g.hue, x=g.x, ax=g.ax_marg_x, orient="h", palette="deep", alpha=0.8
    )
    g.plot_joint(sns.scatterplot, palette="deep")

    # clf= linear_model.LogisticRegression(penalty=None).fit(df[[x,y]],df[hue])
    # x1_plot = np.linspace(*g.ax_joint.get_xlim())
    # x2_plot = np.linspace(*g.ax_joint.get_ylim())
    # xc,yc = np.meshgrid(x1_plot,x2_plot)
    # Z=np.zeros((len(x1_plot),len(x1_plot)))
    # for i in range(len(x1_plot)):
    #     for j in range(len(x2_plot)):
    #         Z[i,j]=clf.decision_function([[xc[i,j],yc[i,j]]])

    # g.ax_joint.contourf(xc,yc,Z,cmap=mpl.colormaps['RdBu'].reversed(), alpha=0.2)
    # print(f'clf auroc = {metrics.roc_auc_score(df[hue],clf.decision_function(df[[x,y]]))}')
    print(pearsonr(df[x], df[y]))
    return g

# %%
g = PlotDecision(All_X.reset_index("focality"), "fivesense", "Unweighted", "focality")
g.figure.tight_layout()
g.figure.subplots_adjust(top=0.92)
g.figure.suptitle("Implant Distance vs. 5-Sense")
h, l = g.ax_joint.get_legend_handles_labels()
g.ax_joint.legend(h, ["nonfocal", "focal"], title="")
g.ax_joint.set_xlabel("5-Sense")
g.ax_joint.set_ylabel("Implant Distance (mm)")
g.savefig(ospj(figpath, "implant_5sense.pdf"))

# %%
g = PlotDecision(All_X.reset_index("focality"), "fivesense", "gamma_bp", "focality")
g.figure.tight_layout()
g.figure.subplots_adjust(top=0.92)
g.figure.suptitle("Gamma Power Abnormality vs. 5-Sense")
h, l = g.ax_joint.get_legend_handles_labels()
g.ax_joint.legend(h, ["nonfocal", "focal"], title="")
g.ax_joint.set_xlabel("5-Sense")
g.ax_joint.set_ylabel("Gamma Power Weighted Distance (mm)")
g.savefig(ospj(figpath, "gamma_5sense.pdf"))

# %%
X = All_X.dropna().reset_index(
    "focality"
)  # .drop('fivesense',axis=1).join(FiveSensePreds.droplevel('focality'))
y = X.pop("focality")

# %%
res, ax = PrettyPictures(X, y, scale=True)
# res['cohend'].abs()

# %%
res.xs("cohend", level="metric").rename(columns={"focality": "Cohen's d"})[
    "Cohen's d"
].abs().sort_values(ascending=False).plot.bar()
plt.ylabel("Cohen's d")
plt.xlabel("Feature")
plt.savefig(ospj(figpath, "cohend.pdf"), bbox_inches="tight")

# %%
order = [
    "fivesense",
    "gamma_bp",
    "broad_bp",
    "delta_bp",
    "theta_bp",
    "alpha_bp",
    "beta_bp",
    "delta_coh",
    "alpha_coh",
    "theta_coh",
    "broad_coh",
    "gamma_coh",
    "beta_coh",
    "Unweighted",
]

# %%
bardata = res.xs("roc", level="metric").rename(columns={"focality": "AUC"})
xpos = [1] + list(np.arange(6) + 3) + list(np.arange(6) + 10) + [17]
error = (
    res.xs("ci", level="metric")["focality"]
    .apply(pd.Series)
    .sub(res.xs("roc", level="metric")["focality"], axis=0)
    .max(axis=1)
)

# %%
# create an errorbar plot of the auc and ci for each feature and use the order above
plt.figure()
plt.bar(
    x=xpos,
    height=res.xs("roc", level="metric").loc[order].values.flatten(),
    yerr=res.xs("ci", level="metric")["focality"]
    .apply(pd.Series)
    .sub(res.xs("roc", level="metric")["focality"], axis=0)
    .loc[order]
    .max(axis=1)
    .values.flatten(),
    tick_label=order,
    color="tab:gray",
)
plt.xticks(rotation=30)
plt.ylabel("AUC (95% CI)")
plt.savefig(ospj(figpath, "univar_auc.pdf"), bbox_inches="tight")

# %%
res, ax = PrettyPictures(
    X[["fivesense", "Unweighted", "gamma_bp", "delta_coh"]], y, scale=True
)
h, l = ax.get_legend_handles_labels()
ax.legend(h, ["nonfocal", "Focal"], title=None, loc=(1.02, 0.4))
ax.set_xticklabels(["5-Sense", "Implant Distance", "Gamma Power", "Delta Coherence"])
plt.xticks(rotation=0)
plt.xlabel("")
plt.ylabel("scaled value")
plt.ylim((-0.08, 1.19))
plt.savefig(ospj(figpath, "univariate.pdf"), bbox_inches="tight")

# %%
g = sns.clustermap(
    pd.DataFrame(
        preprocessing.StandardScaler().fit_transform(X),
        index=X.index,
        columns=X.columns,
    )
    .corr()
    .abs(),
    annot=True,
    figsize=(10, 10),
)
g.savefig(ospj(figpath, "clustermap.pdf"), bbox_inches="tight")

plt.savefig(ospj(figpath, "clustermap.pdf"))

# %%
X_subsets = {
    "Preimplant Baseline": All_X[["fivesense", "Unweighted"]],
    "IEEG Model": All_X,  # for supplement: remove comment this line
}

# %%
models = {}
for k, X in X_subsets.items():
    X = X.reset_index("focality")
    y_all = X.pop("focality")
    print(k, X.shape)
    in_sample = MakePredictions(X, y, k)
    print(metrics.roc_auc_score(y, in_sample))

    models[k] = ML_Model(X, y_all)
pd.concat(
    [pd.Series(mod[0]) for mod in models.values()], keys=[key for key in models.keys()]
)

# %%
all_tprs = []
for k, v in models.items():
    all_tprs.append(v[1])
roc_curves = (
    pd.concat(all_tprs, keys=models.keys(), names=["model", "fpr"])
    .melt(ignore_index=False, var_name="fold", value_name="tpr")
    .reset_index()
)
fig, ax = plt.subplots()
sns.lineplot(
    data=roc_curves,
    x="fpr",
    y="tpr",
    hue="model",
    ax=ax,
    hue_order=["IEEG Model", "Preimplant Baseline"],
)
plt.plot([0, 1], [0, 1], color="tab:gray", linestyle="--")
plt.ylabel("Sensitivity (correctly predicting focality)")
plt.xlabel("1-Specificity (incorrectly predicting focality)")
fpr, tpr, thresh = metrics.roc_curve(y_all.values, All_X["fivesense"].values)
plt.plot(fpr, tpr, label="5-Sense")
h, l = ax.get_legend_handles_labels()
plt.legend(
    h, ["IEEG Model ROC=", "Implant Baseline ROC=", "5-Sense ROC="], loc="lower right"
)
plt.savefig(ospj(figpath, "roc_curve.pdf"), bbox_inches="tight")

# %%
roc_curves["distance"] = np.sqrt(roc_curves["fpr"] ** 2 + (1 - roc_curves["tpr"]) ** 2)

# %%
coefs = models["IEEG Model"][3]
coefs = coefs.loc[:, ~(coefs == 0).all()].abs()
order = coefs.mean().sort_values(ascending=False).index.tolist()
sns.barplot(coefs, order=order, orient="h", errorbar="sd", color="tab:blue")
plt.xlabel("Coefficient, mean (sd)")
plt.savefig(ospj(figpath, "coefs1.pdf"), bbox_inches="tight")

# %%
PredClass = []
for k, mod in models.items():
    PredClass.append(
        mod[2]["true"]
        .to_frame()
        .join(
            pd.Series(mod[2]["yhat"], index=mod[2]["true"].index).rename(
                f"{k} Prediction"
            )
        )
        .set_index("focality", append=True)
    )

# %%
models

# %%
PredClass = (
    pd.concat(PredClass, axis=1)
    .groupby(["patient", "focality"])
    .mean()
    .join((FiveSensePreds > 0.376).astype(int), how="inner")
    .reset_index("focality")
)

# %%
yhat = FiveSensePreds.loc[All_Patient] > 0.376
print(metrics.classification_report(yhat.index.get_level_values("focality"), yhat))

# %%
print(
    metrics.classification_report(
        PredClass["focality"], PredClass["IEEG Model Prediction"]
    )
)

# %%
pd.crosstab(PredClass["focality"], PredClass["IEEG Model Prediction"])

# %%
pd.crosstab(PredClass["focality"], PredClass["fivesense"])

# %%
# return true where other columns of predclass are equal to predclass['focality']
PredClass["IEEG Model Prediction"] = (
    PredClass["IEEG Model Prediction"] == PredClass["focality"]
).map({True: "correct", False: "incorrect"})
PredClass["fivesense"] = (PredClass["fivesense"] == PredClass["focality"]).map(
    {True: "correct", False: "incorrect"}
)

# %%
pd.crosstab(PredClass["fivesense"], PredClass["IEEG Model Prediction"])

# %%
PredClass.query(
    'fivesense == "incorrect" and `IEEG Model Prediction` == "correct"'
).join(PreImplantData["RID"])

# %%
PredClass.query(
    'fivesense == "correct" and `IEEG Model Prediction` == "incorrect"'
).join(PreImplantData["RID"])

# %%
from tools import delong

assert (
    models["Preimplant Baseline"][2]["true"] == models["IEEG Model"][2]["true"]
).all()
AvgPreds = []
for k, mod in models.items():
    AvgPreds.append(
        mod[2]["true"]
        .to_frame()
        .join(
            pd.Series(mod[2]["pred"], index=mod[2]["true"].index).rename(
                f"{k} Prediction"
            )
        )
        .set_index("focality", append=True)
    )

# %%
AvgPreds = (
    pd.concat(AvgPreds, axis=1)
    .groupby(["patient", "focality"])
    .mean()
    .join(FiveSensePreds, how="inner")
    .reset_index("focality")
)

# %%
print(
    10
    ** delong.delong_roc_test(
        AvgPreds["focality"], AvgPreds["IEEG Model Prediction"], AvgPreds["fivesense"]
    )
)
print(
    10
    ** delong.delong_roc_test(
        AvgPreds["focality"],
        AvgPreds["IEEG Model Prediction"],
        AvgPreds["Preimplant Baseline Prediction"],
    )
)
print(
    10
    ** delong.delong_roc_test(
        AvgPreds["focality"],
        AvgPreds["fivesense"],
        AvgPreds["Preimplant Baseline Prediction"],
    )
)

# %%
sm.stats.multipletests(
    [
        10
        ** delong.delong_roc_test(
            AvgPreds["focality"],
            AvgPreds["IEEG Model Prediction"],
            AvgPreds["fivesense"],
        )[0][0],
        10
        ** delong.delong_roc_test(
            AvgPreds["focality"],
            AvgPreds["IEEG Model Prediction"],
            AvgPreds["Preimplant Baseline Prediction"],
        )[0][0],
        10
        ** delong.delong_roc_test(
            AvgPreds["focality"],
            AvgPreds["fivesense"],
            AvgPreds["Preimplant Baseline Prediction"],
        )[0][0],
    ],
    method="holm",
)

# %%
fpr, tpr, thresh = metrics.roc_curve(
    AvgPreds["focality"], AvgPreds["IEEG Model Prediction"]
)

# %%
roc_curves = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresh": thresh})
roc_curves["distance"] = np.sqrt(roc_curves["fpr"] ** 2 + (1 - roc_curves["tpr"]) ** 2)
plt.plot(roc_curves["fpr"], roc_curves["tpr"], label="IEEG Model")

# %%
X = AvgPreds.copy()
y = X.pop("focality")

# %%
metric_scores, ax = PrettyPictures(
    X[["fivesense", "Preimplant Baseline Prediction", "IEEG Model Prediction"]], y
)
ax.set_xlabel("")
h, l = ax.get_legend_handles_labels()
ax.legend(h, ["nonfocal", "focal"], loc=(1.02, 0.4))
ax.set_xticklabels(["5-Sense", "Implant Baseline", "IEEG Model"])
ax.set_ylim(-0.05, 1.15)
plt.xticks(
    rotation=0,
)
plt.savefig(ospj(figpath, "ieegmodel.pdf"), bbox_inches="tight")

# %%
with sns.axes_style("ticks"):
    f, g = PlotLogit(X, y)
    h, l = f.ax_joint.get_legend_handles_labels()
    f.ax_joint.legend(handles=h, labels=["Nonfocal", "Focal"], loc="best")
    f.figure.savefig(ospj(figpath, "fivesense_corr.pdf"), bbox_inches="tight")
    # f.ax_joint.spines[:].set_visible(True)
    h, l = g.ax_joint.get_legend_handles_labels()
    g.ax_joint.legend(handles=h, labels=["Nonfocal", "Focal"], loc="best")
    g.figure.savefig(ospj(figpath, "null_corr.pdf"), bbox_inches="tight")

# %%
# data = pd.DataFrame({'logit proba':preds,
#                   'unweighted':preds_unweighted,
#                     'focality':y
#                  })
# g = sns.lmplot(data, x = 'unweighted', y = 'logit proba', hue = 'focality', fit_reg= False)
# sns.regplot(data, x = 'unweighted', y = 'logit proba', scatter=False, ax=g.axes[0, 0])
# plt.ylabel('logit proba')
# plt.text(0, 0.45, 'r = {:0.2f}'.format(np.corrcoef(preds, preds_unweighted)[0,1]))

# %% [markdown]
# ### Analyze outcomes

# %%
exog = X.join(
    type_of_proc[type_of_proc.isin(["surgery", "device"])].rename("procedure_type"),
    how="inner",
)
endog = exog.pop("procedure_type").map({"surgery": 1, "device": 0})
print(endog.value_counts())
score, ax = PrettyPictures(
    exog[["fivesense", "Preimplant Baseline Prediction", "IEEG Model Prediction"]],
    endog,
)
metric_scores = metric_scores.join(score)
ax.set_xlabel("")
h, l = ax.get_legend_handles_labels()
ax.legend(h, l, title="", loc=(1.02, 0.4))
ax.set_xticklabels(["5-Sense", "Preimplant Baseline", "IEEG Model"])
ax.set_ylim(-0.05, 1.15)
plt.xticks(rotation=0)
plt.savefig(ospj(figpath, "outcomes_surgery.pdf"), bbox_inches="tight")
# PlotLogit(exog,endog)

# %%
PlotSurgData = (
    exog[["fivesense", "Preimplant Baseline Prediction", "IEEG Model Prediction"]]
    .melt(ignore_index=False, var_name="variable")
    .join(endog, how="inner")
)
plt.figure()
ax = sns.boxplot(
    data=PlotSurgData,
    x="variable",
    y="value",
    hue="procedure_type",
    palette=sns.color_palette(["#a8ddb5", "#9ecae1"]),
    showfliers=False,
)
sns.stripplot(
    data=PlotSurgData,
    x="variable",
    y="value",
    hue="procedure_type",
    palette=sns.color_palette(["#608269", "#08519c"]),
    alpha=0.8,
    dodge=True,
    legend=False,
    ax=ax,
)

ax.set_xlabel("")
h, l = ax.get_legend_handles_labels()
ax.legend(h, l, title="", loc=(1.02, 0.4))
ax.set_xticklabels(["5-Sense", "Preimplant Baseline", "IEEG Model"])
ax.set_ylim(-0.05, 1.15)
plt.xticks(rotation=0)
plt.ylabel("probability focal")
plt.savefig(ospj(figpath, "outcomes_surgery.pdf"), bbox_inches="tight")

# %%
outcomes

# %%
ILAE = (
    outcomes.loc[((type_of_proc == "surgery"))][
        [
            "1 year",
            "2 years",
        ]
    ]
    .replace("ILAE 1a", "ILAE 1")
    .astype("category")
)
ILAE = (
    ILAE.applymap(lambda x: int(str(x).strip()[-1]) if x is not np.nan else np.nan)
    .mask(lambda x: x.isin([1, 2]), 1)
    .where(lambda x: x.isin([1, np.nan]), 0)
)
engel = outcomes.loc[((type_of_proc == "surgery"))][
    [
        "1 year.1",
        "2 years.1",
    ]
]
assert (
    not (
        engel.applymap(lambda x: str(x).strip()[-1] if x is not np.nan else np.nan)
        == "I"
    )
    .any()
    .any()
)
engel = engel.applymap(
    lambda x: str(x).strip()[:-1].split(" ")[-1] if x is not np.nan else np.nan
)
engel = engel.replace(["I", "II", "III", "IV"], [1, 2, 3, 4]).where(
    lambda x: x.isin([1, np.nan]), 0
)
Surg_Outcomes = engel.join(ILAE)

# %%
Surg_Outcomes

# %%
# outcome_classes = np.unique(outcomes[((type_of_proc=='surgery'))][["1 year", "1 year.1", "2 years", "2 years.1"]].dropna().values.ravel().astype(str))
# Surg_Outcomes = (
#     outcomes[( (type_of_proc == "surgery"))][
#         ["1 year", "1 year.1", "2 years", "2 years.1"]
#     ]
#     .mask(
#         lambda x: x.isin(
#             [
#                 "Engel class IA",
#                 "Engel class IB",
#                 "Engel class IC",
#                 "Engel class ID",
#                 "ILAE 1",
#                 "ILAE 1a",
#                 # "ILAE 2",
#             ]
#         ),
#         1,
#     )
#     .mask(lambda x: x.isin(outcome_classes), 0)
# )
# Surg_Outcomes.value_counts(dropna=False)

# %%
GoodOutcome1 = (
    (Surg_Outcomes["1 year.1"].dropna() == 1)
    # |
    # (Surg_Outcomes["1 year"].dropna() == 1)
).rename("good_outcome_1yr")

# %%
exog = X.join(GoodOutcome1).dropna().copy()
endog = exog.pop("good_outcome_1yr").astype(int)
print(endog.value_counts())
score, ax = PrettyPictures(exog, endog)
metric_scores = metric_scores.join(score)
# PlotLogit(exog,endog)

# %%
exog = pd.concat(
    [
        X[X.index.isin(GoodOutcome1[~GoodOutcome1].index)],
        X[X.index.isin(type_of_proc[type_of_proc == "device"].index)],
    ],
    keys=["relapse", "device"],
    names=["relapse_1_device"],
).reset_index("relapse_1_device")
endog = exog.pop("relapse_1_device").map({"relapse": 1, "device": 0})
print(endog.value_counts())
score, ax = PrettyPictures(exog, endog)
metric_scores = metric_scores.join(score)
# MakePredictions(exog.drop('all_data',axis=1),endog.map({'relapse':1, 'device':0}))
# PlotLogit(exog, endog)

# %%
exog = pd.concat(
    [
        X[X.index.isin(GoodOutcome1[GoodOutcome1].index)],
        X[X.index.isin(type_of_proc[type_of_proc == "device"].index)],
    ],
    keys=["surg_good", "device"],
    names=["szfree_1_device"],
).reset_index("szfree_1_device")
endog = exog.pop("szfree_1_device").map({"surg_good": 1, "device": 0})
print(endog.value_counts())
score, ax = PrettyPictures(exog, endog)
metric_scores = metric_scores.join(score)
# MakePredictions(exog.drop('all_data',axis=1),endog.map({'surg_good':1, 'device':0}))
# PlotLogit(exog, endog)

# %%
GoodOutcome1 = (
    # (Surg_Outcomes["1 year.1"].dropna() == 1)
    # |
    (Surg_Outcomes["1 year"].dropna() == 1)
).rename("good_outcome_1yr_ilae")

# %%
exog = X.join(GoodOutcome1).dropna().copy()
endog = exog.pop("good_outcome_1yr_ilae").astype(int)
print(endog.value_counts())
score, ax = PrettyPictures(exog, endog)
metric_scores = metric_scores.join(score)
# PlotLogit(exog,endog)

# %%
exog = pd.concat(
    [
        X[X.index.isin(GoodOutcome1[~GoodOutcome1].index)],
        X[X.index.isin(type_of_proc[type_of_proc == "device"].index)],
    ],
    keys=["relapse", "device"],
    names=["relapse_1_device_ilae"],
).reset_index("relapse_1_device_ilae")
endog = exog.pop("relapse_1_device_ilae").map({"relapse": 1, "device": 0})
print(endog.value_counts())
score, ax = PrettyPictures(exog, endog)
metric_scores = metric_scores.join(score)
# MakePredictions(exog.drop('all_data',axis=1),endog.map({'relapse':1, 'device':0}))
# PlotLogit(exog, endog)

# %%
exog = pd.concat(
    [
        X[X.index.isin(GoodOutcome1[GoodOutcome1].index)],
        X[X.index.isin(type_of_proc[type_of_proc == "device"].index)],
    ],
    keys=["surg_good", "device"],
    names=["szfree_1_device_ilae"],
).reset_index("szfree_1_device_ilae")
endog = exog.pop("szfree_1_device_ilae").map({"surg_good": 1, "device": 0})
print(endog.value_counts())
score, ax = PrettyPictures(exog, endog)
metric_scores = metric_scores.join(score)
# MakePredictions(exog.drop('all_data',axis=1),endog.map({'surg_good':1, 'device':0}))
# PlotLogit(exog, endog)

# %%
GoodOutcome2 = (
    (Surg_Outcomes["2 years.1"].dropna() == 1)
    # &
    # (Surg_Outcomes["2 years"].dropna() == 1)
    # & (GoodOutcome1)
).rename("good_outcome_2yr")

# %%
exog = X.join(GoodOutcome2, how="inner").dropna().copy()
endog = exog.pop("good_outcome_2yr").astype(int)
print(endog.value_counts())
score, ax = PrettyPictures(
    exog[["fivesense", "Preimplant Baseline Prediction", "IEEG Model Prediction"]],
    endog,
)
metric_scores = metric_scores.join(score)
ax.set_xlabel("")
h, l = ax.get_legend_handles_labels()
ax.set_xticklabels(["5-Sense", "Preimplant Baseline", "IEEG Model"])
plt.xticks(rotation=0)
# PlotLogit(exog,endog)

# %%
exog = pd.concat(
    [
        X[X.index.isin(GoodOutcome2[~GoodOutcome2].index)],
        X[X.index.isin(type_of_proc[type_of_proc == "device"].index)],
    ],
    keys=["relapse", "device"],
    names=["relapse_2_device"],
).reset_index("relapse_2_device")
endog = exog.pop("relapse_2_device").map({"relapse": 1, "device": 0})
print(endog.value_counts())
score, ax = PrettyPictures(exog, endog)
metric_scores = metric_scores.join(score)
# MakePredictions(exog.drop('all_data',axis=1),endog.map({'relapse':1, 'device':0}))
# PlotLogit(exog, endog)

# %%
exog = pd.concat(
    [
        X[X.index.isin(GoodOutcome2[GoodOutcome2].index)],
        X[X.index.isin(type_of_proc[type_of_proc == "device"].index)],
    ],
    keys=["surg_good", "device"],
    names=["szfree_2_device"],
).reset_index("szfree_2_device")
endog = exog.pop("szfree_2_device").map({"surg_good": 1, "device": 0})
print(endog.value_counts())
score, ax = PrettyPictures(exog, endog)
metric_scores = metric_scores.join(score)
# MakePredictions(exog.drop('all_data',axis=1),endog.map({'surg_good':1, 'device':0}))
# PlotLogit(exog, endog)

# %%
GoodOutcome2 = (
    # (Surg_Outcomes["2 years.1"].dropna() == 1)
    # &
    (Surg_Outcomes["2 years"].dropna() == 1)
    # & (GoodOutcome1)
).rename("good_outcome_2yr_ilae")

# %%
exog = X.join(GoodOutcome2, how="inner").dropna().copy()
endog = exog.pop("good_outcome_2yr_ilae").astype(int)
print(endog.value_counts())
score, ax = PrettyPictures(
    exog[["fivesense", "Preimplant Baseline Prediction", "IEEG Model Prediction"]],
    endog,
)
metric_scores = metric_scores.join(score)
ax.set_xlabel("")
h, l = ax.get_legend_handles_labels()
ax.set_xticklabels(["5-Sense", "Preimplant Baseline", "IEEG Model"])
plt.xticks(rotation=0)
# PlotLogit(exog,endog)

# %%
exog = pd.concat(
    [
        X[X.index.isin(GoodOutcome2[~GoodOutcome2].index)],
        X[X.index.isin(type_of_proc[type_of_proc == "device"].index)],
    ],
    keys=["relapse", "device"],
    names=["relapse_2_device_ilae"],
).reset_index("relapse_2_device_ilae")
endog = exog.pop("relapse_2_device_ilae").map({"relapse": 1, "device": 0})
print(endog.value_counts())
score, ax = PrettyPictures(exog, endog)
metric_scores = metric_scores.join(score)
# MakePredictions(exog.drop('all_data',axis=1),endog.map({'relapse':1, 'device':0}))
# PlotLogit(exog, endog)

# %%
exog = pd.concat(
    [
        X[X.index.isin(GoodOutcome2[GoodOutcome2].index)],
        X[X.index.isin(type_of_proc[type_of_proc == "device"].index)],
    ],
    keys=["surg_good", "device"],
    names=["szfree_2_device_ilae"],
).reset_index("szfree_2_device_ilae")
endog = exog.pop("szfree_2_device_ilae").map({"surg_good": 1, "device": 0})
print(endog.value_counts())
score, ax = PrettyPictures(exog, endog)
metric_scores = metric_scores.join(score)
# MakePredictions(exog.drop('all_data',axis=1),endog.map({'surg_good':1, 'device':0}))
# PlotLogit(exog, endog)

# %%
GoodOutcome = pd.merge(
    GoodOutcome1, GoodOutcome2, how="outer", left_index=True, right_index=True
)

# %%
metric_scores

# %%
outcome_groups = [
    [
        "good_outcome_1yr",
        "relapse_1_device",
        "szfree_1_device",
    ],
    ["good_outcome_1yr_ilae", "relapse_1_device_ilae", "szfree_1_device_ilae"],
    [
        "good_outcome_2yr",
        "relapse_2_device",
        "szfree_2_device",
    ],
    ["good_outcome_2yr_ilae", "relapse_2_device_ilae", "szfree_2_device_ilae"],
]

# %%
unique_categories = metric_scores.index.get_level_values("variable").unique()
for cat in unique_categories:
    for group in outcome_groups:
        print(cat, group)
        print(
            metric_scores.xs("ranksum", level="metric").loc[cat, group],
        )
        print(
            sm.stats.multipletests(
                metric_scores.xs("ranksum", level="metric").loc[cat, group],
                method="holm",
            )
        )

# %%
sns.heatmap(
    metric_scores.xs("ranksum", level="metric") < 0.05,
    annot=metric_scores.xs("roc", level="metric"),
    annot_kws={"size": 10},
)

# %%
metric_scores

# %%
GoodOutcome2 = (
    # (Surg_Outcomes["2 years.1"].dropna() == 1)
    # &
    (Surg_Outcomes["2 years"].dropna() == 1)
    # & (GoodOutcome1)
).rename("good_outcome_2yr")

# %%
Fig2Data = pd.concat(
    [
        X.join(
            GoodOutcome2.map({True: "good outcome", False: "poor outcome"}).rename(
                "group"
            ),
            how="inner",
        ),
        X.join(
            type_of_proc[type_of_proc.isin(["device"])].rename("group"), how="inner"
        ),
    ],
    keys=["proc_type", "outcome"],
    names=["sep"],
).reset_index("sep")
print(Fig2Data["group"].value_counts())

plt.figure(figsize=(4, 6))
ax = sns.boxplot(
    data=Fig2Data,
    x="group",
    y="IEEG Model Prediction",
    palette=sns.color_palette(["#e0f3db", "#a8ddb5", "#9ecae1"]),
    order=["good outcome", "poor outcome", "device"],
    width=0.5,
    showfliers=False,
)
for art in ax.patches:
    r, g, b, a = art.get_facecolor()
    art.set_facecolor((r, g, b, 0.7))
sns.swarmplot(
    data=Fig2Data,
    x="group",
    y="IEEG Model Prediction",
    palette=sns.color_palette(["#65a355", "#608269", "#08519c"]),
    order=["good outcome", "poor outcome", "device"],
)

plt.xlabel("")
plt.ylim(-0.04, 1.15)
plt.yticks(np.linspace(0, 1, 6))
plt.ylabel("IEEG Model")
ticks, lbls = plt.xticks()
plt.xticks(ticks, ["ILAE 1 or 2", "ILAE 3+", "device"])
plt.savefig(ospj(figpath, "outcomes_ieeg_ilae.pdf"), bbox_inches="tight")
fig.show()

# %%
plt.figure(figsize=(4, 6))
ax = sns.boxplot(
    data=Fig2Data,
    x="group",
    y="fivesense",
    palette=sns.color_palette(["#e0f3db", "#a8ddb5", "#9ecae1"]),
    order=["good outcome", "poor outcome", "device"],
    width=0.5,
    showfliers=False,
)
for art in ax.patches:
    r, g, b, a = art.get_facecolor()
    art.set_facecolor((r, g, b, 0.7))
sns.swarmplot(
    data=Fig2Data,
    x="group",
    y="fivesense",
    palette=sns.color_palette(["#65a355", "#608269", "#08519c"]),
    order=["good outcome", "poor outcome", "device"],
    alpha=0.8,
)
plt.xlabel("")
plt.ylim(-0.04, 1.15)
plt.ylabel("Post Test Probability")
plt.ylabel("5-Sense")
ticks, lbls = plt.xticks()
plt.xticks(ticks, ["ILAE 1 or 2", "ILAE 3+", "device"])
plt.savefig(ospj(figpath, "outcomes_5sense_ilae.pdf"), bbox_inches="tight")
plt.show()

# %%
plt.figure(figsize=(4, 6))
Ax = sns.boxplot(
    data=Fig2Data,
    x="group",
    y="Preimplant Baseline Prediction",
    palette=sns.color_palette(["#e0f3db", "#a8ddb5", "#9ecae1"]),
    order=["good outcome", "poor outcome", "device"],
    width=0.5,
    showfliers=False,
)
for art in ax.patches:
    r, g, b, a = art.get_facecolor()
    art.set_facecolor((r, g, b, 0.7))
sns.swarmplot(
    data=Fig2Data,
    x="group",
    y="Preimplant Baseline Prediction",
    palette=sns.color_palette(["#65a355", "#608269", "#08519c"]),
    order=["good outcome", "poor outcome", "device"],
    alpha=0.8,
)
plt.ylabel("Implant Baseline")
plt.ylim(-0.04, 1.15)
plt.yticks(np.linspace(0, 1, 6))
plt.xlabel("")
ticks, lbls = plt.xticks()
plt.xticks(ticks, ["ILAE 1 or 2", "ILAE 3+", "device"])
plt.savefig(ospj(figpath, "outcomes_baseline_ilae.pdf"), bbox_inches="tight")
plt.show()

# %%
GoodOutcome2 = (
    (Surg_Outcomes["2 years.1"].dropna() == 1)
    # &
    # (Surg_Outcomes["2 years"].dropna() == 1)
    # & (GoodOutcome1)
).rename("good_outcome_2yr")

# %%
Fig2Data = pd.concat(
    [
        X.join(
            GoodOutcome2.map({True: "good outcome", False: "poor outcome"}).rename(
                "group"
            ),
            how="inner",
        ),
        X.join(
            type_of_proc[type_of_proc.isin(["device"])].rename("group"), how="inner"
        ),
    ],
    keys=["proc_type", "outcome"],
    names=["sep"],
).reset_index("sep")
print(Fig2Data["group"].value_counts())

plt.figure(figsize=(4, 6))
ax = sns.boxplot(
    data=Fig2Data,
    x="group",
    y="IEEG Model Prediction",
    palette=sns.color_palette(["#e0f3db", "#a8ddb5", "#9ecae1"]),
    order=["good outcome", "poor outcome", "device"],
    width=0.5,
    showfliers=False,
)
for art in ax.patches:
    r, g, b, a = art.get_facecolor()
    art.set_facecolor((r, g, b, 0.7))
sns.swarmplot(
    data=Fig2Data,
    x="group",
    y="IEEG Model Prediction",
    palette=sns.color_palette(["#65a355", "#608269", "#08519c"]),
    order=["good outcome", "poor outcome", "device"],
)

plt.xlabel("")
plt.ylim(-0.04, 1.15)
plt.yticks(np.linspace(0, 1, 6))
plt.ylabel("IEEG Model")
ticks, lbls = plt.xticks()
plt.xticks(ticks, ["Engel 1", "Engel 2+", "device"])
plt.savefig(ospj(figpath, "outcomes_ieeg.pdf"), bbox_inches="tight")
fig.show()

# %%
plt.figure(figsize=(4, 6))
ax = sns.boxplot(
    data=Fig2Data,
    x="group",
    y="fivesense",
    palette=sns.color_palette(["#e0f3db", "#a8ddb5", "#9ecae1"]),
    order=["good outcome", "poor outcome", "device"],
    width=0.5,
    showfliers=False,
)
for art in ax.patches:
    r, g, b, a = art.get_facecolor()
    art.set_facecolor((r, g, b, 0.7))
sns.swarmplot(
    data=Fig2Data,
    x="group",
    y="fivesense",
    palette=sns.color_palette(["#65a355", "#608269", "#08519c"]),
    order=["good outcome", "poor outcome", "device"],
    alpha=0.8,
)
plt.xlabel("")
plt.ylim(-0.04, 1.15)
plt.ylabel("Post Test Probability")
plt.ylabel("5-Sense")
ticks, lbls = plt.xticks()
plt.xticks(ticks, ["Engel 1", "Engel 2+", "device"])
plt.savefig(ospj(figpath, "outcomes_5s.pdf"), bbox_inches="tight")
plt.show()

# %%
plt.figure(figsize=(4, 6))
Ax = sns.boxplot(
    data=Fig2Data,
    x="group",
    y="Preimplant Baseline Prediction",
    palette=sns.color_palette(["#e0f3db", "#a8ddb5", "#9ecae1"]),
    order=["good outcome", "poor outcome", "device"],
    width=0.5,
    showfliers=False,
)
for art in ax.patches:
    r, g, b, a = art.get_facecolor()
    art.set_facecolor((r, g, b, 0.7))
sns.swarmplot(
    data=Fig2Data,
    x="group",
    y="Preimplant Baseline Prediction",
    palette=sns.color_palette(["#65a355", "#608269", "#08519c"]),
    order=["good outcome", "poor outcome", "device"],
    alpha=0.8,
)
plt.ylabel("Implant Baseline")
plt.ylim(-0.04, 1.15)
plt.yticks(np.linspace(0, 1, 6))
plt.xlabel("")
ticks, lbls = plt.xticks()
plt.xticks(ticks, ["Engel 1", "Engel 2+", "device"])
plt.savefig(ospj(figpath, "outcomes_baseline.pdf"), bbox_inches="tight")
plt.show()

# %%
Fig2Data["group"].value_counts()

# %%
## here differentiate between laser or resection

all_procs[
    [
        "Laser ablation",
        "Resection with intracranial implant",
        "Resection without intracranial implant",
    ]
].value_counts()
ablation_resection = all_procs[
    ["Laser ablation"] + [col for col in all_procs.columns if "Resection" in col]
].apply(
    lambda x: (
        "both"
        if x.sum() > 1
        else (
            "ablation"
            if x["Laser ablation"]
            else (
                "resection"
                if x["Resection with intracranial implant"]
                | x["Resection without intracranial implant"]
                else 0
            )
        )
    ),
    axis=1,
)

# %% [markdown]
# #### Make Table

# %%
N_chan_lat = (
    (
        AllBandChans.reset_index()
        .merge(LUT, left_on="label", right_on="roi_DKT")[
            ["patient", "focality", "channel", "x_mm", "y_mm", "z_mm", "lat"]
        ]
        .drop_duplicates()
    )
    .groupby(["patient", "focality"])["lat"]
    .value_counts()
    .unstack("lat")
    .droplevel("focality")
)
N_chan_lat["tot"] = N_chan_lat.sum(axis=1)
N_chan_lat["Implant"] = (
    N_chan_lat[["left", "right"]]
    .notna()
    .all(axis=1)
    .map({True: "Bilateral", False: "Unilateral"})
)

# %%
x1 = N_chan_lat[["tot"]].join(Y)
y1 = x1.pop("focality")

# %%
PrettyPictures(x1, y1)

# %%
SelectPatient = (
    X.join(Y).join(type_of_proc.rename("proc_type")).join(GoodOutcome).join(N_chan_lat)
)

# %%
from glob import glob

# %%
# for pt in tqdm(All_X.index.get_level_values("patient").values):
#     borel_path = "/users/rsg20/ieeg_recon/BIDS"
#     pt_subfolder = "sub-RID0" + f"{PreImplantData.loc[pt].RID:03n}"
#     write_folder = ospj(borel_path, pt_subfolder, "plots")
#     Pt_Electrode_ZScores = []
#     Feat_Names = []
#     pt_data = All_X.loc[pt]
#     for feat_name, val in pt_data.items():
#         if "_bp" in feat_name:
#             band = [b for b in band_names if b in feat_name][0]
#             clip_num = (
#                 BP_WSD_all_clips.loc[pt, band]
#                 .reset_index()
#                 .iloc[(BP_WSD_all_clips.loc[pt, band] - val).abs().to_numpy().argmin()][
#                     "clip"
#                 ]
#             )
#             Pt_Electrode_ZScores.append(
#                 AllBandChans.reset_index()
#                 .query(
#                     '(patient == @pt) & (band == @band) & (clip == @clip_num) & (period == "interictal")'
#                 )[["patient", "channel", "band", "absZScore"]]
#                 .merge(
#                     ChansToGet.set_index(["patient", "name"])[["x_mm", "y_mm", "z_mm"]],
#                     left_on=["patient", "channel"],
#                     right_index=True,
#                 )
#             )
#             Feat_Names.append(feat_name)
#         elif "_coh" in feat_name:
#             band = [b for b in band_names if b in feat_name][0]
#             clip_num = (
#                 Coh_WSD_all_clips.loc[pt, :, band]
#                 .reset_index()
#                 .iloc[
#                     (Coh_WSD_all_clips.loc[pt, :, band] - val).abs().to_numpy().argmin()
#                 ]["clip"]
#             )
#             Pt_Electrode_ZScores.append(
#                 Nxx_Chan_Stats.reset_index()
#                 .query("(patient == @pt) & (band == @band) & (clip == @clip_num)")
#                 .rename({0.75: "absZScore"}, axis=1)[
#                     ["patient", "channel", "band", "absZScore"]
#                 ]
#                 .merge(
#                     ChansToGet.set_index(["patient", "name"])[["x_mm", "y_mm", "z_mm"]],
#                     left_on=["patient", "channel"],
#                     right_index=True,
#                 )
#             )
#             Feat_Names.append(feat_name)
#         PtChanData = (
#             pd.concat(Pt_Electrode_ZScores, keys=Feat_Names, names=["Features"])
#             .droplevel(-1)
#             .reset_index()
#         )
#         PtChanData.to_csv(ospj(data_path, pt, "plot_abnormality.csv"))
#     for feat, feat_score in PtChanData.groupby("Features"):
#         node_file = feat_score.loc[:, ["x_mm", "y_mm", "z_mm", "absZScore"]]
#         node_file["absZScore"] = node_file["absZScore"].mask(lambda x: x > 3.5, 3.5)
#         node_file["size"] = node_file["absZScore"]
#         node_file["label"] = feat_score["channel"]
#         node_file = node_file[["x_mm", "y_mm", "z_mm", "absZScore", "size", "label"]]
#         if not os.path.exists(write_folder):
#             os.mkdir(write_folder)
#         node_file.to_csv(
#             ospj(write_folder, f"{feat}.node"),
#             sep=" ",
#             header=False,
#             index=False,
#             index_label=False,
#         )

# %%
Pivot = (
    SelectPatient.join(
        (
            outcomes["Type of Implant: "].dropna()
            == "Stereo EEG (Depths ONLY, do not check Depths)"
        )
        .map({True: "stereo", False: "grid/strip/depth"})
        .rename("stereo EEG")
    )
    .join(ablation_resection.rename("abl_res"))
    .join(outcomes[["Gender", "Age of Epilepsy onset (in years)", "Age at implant"]])
    .join(FiveSenseDf[[col for col in FiveSenseDf.columns if "MRI" in col]])
    .join(Surg_Outcomes)
    .reset_index()
)
Pivot

# %%
Pivot["Age at implant"]

# %%
Pivot.pivot_table(
    columns="focality",
    index=["proc_type"],
    values="patient",
    aggfunc="count",
    dropna=False,
    fill_value=0,
    margins=True,
)

# %%
Pivot.pivot_table(
    columns="focality",
    index=[
        "stereo EEG",
        "proc_type",
    ],
    values="patient",
    aggfunc="count",
    dropna=False,
    fill_value=0,
    margins=True,
)

# %%
Pivot.columns

# %%
print(Pivot.groupby("focality")["tot"].describe())
mannwhitneyu(
    *[group.dropna().to_numpy() for _, group in Pivot.groupby("focality")["tot"]]
)

# %%
from scipy.stats import chi2_contingency

# %%
chi2_contingency([[4, 33], [5, 29]])

# %%
# chi2_contingency(
Pivot.pivot_table(
    columns="focality",
    index=[
        "2 years.1",
    ],
    values="patient",
    aggfunc="count",
    dropna=False,
    # fill_value=0,
    # margins=True,
)
# )

# %%
Pivot.pivot_table(
    columns="focality",
    index=["stereo EEG", "abl_res", "good_outcome_2yr_ilae"],
    values="patient",
    aggfunc="count",
    dropna=False,
    fill_value=0,
    margins=True,
)

# %%
sns.swarmplot(
    data=Pivot.set_index("patient").join(
        AllBandChans.reset_index()[["patient", "channel", "soz"]]
        .drop_duplicates()
        .groupby("patient")["channel"]
        .count()
        .rename("n_chans")
    ),
    x="stereo EEG",
    y="n_chans",
    hue="focality",
    dodge=True,
    legend=False,
)
sns.boxplot(
    data=Pivot.set_index("patient").join(
        AllBandChans.reset_index()[["patient", "channel", "soz"]]
        .drop_duplicates()
        .groupby("patient")["channel"]
        .count()
        .rename("n_chans")
    ),
    x="stereo EEG",
    y="n_chans",
    hue="focality",
)

# %% [markdown]
# #### Subanalysis on grids/strips vs stereo

# %%
outcomes["seeg"] = (
    outcomes["Type of Implant: "].dropna()
    == "Stereo EEG (Depths ONLY, do not check Depths)"
)

# %%
X.join(outcomes["Type of Implant: "])["Type of Implant: "].value_counts()

# %%
GridStereo = (
    X.join(
        (
            outcomes["Type of Implant: "].dropna()
            == "Stereo EEG (Depths ONLY, do not check Depths)"
        )
        .map({False: "grid/strip/depth", True: "seeg"})
        .rename("stereo EEG"),
        how="inner",
    )
    .join(y)
    .join(Base_SD.droplevel("focality"), how="inner")
)
# GridStereo['stereo EEG']=GridStereo['stereo EEG'].map({True:'seeg', False:'grid/strip/depth'})

# %%
GridStereo

# %%
plt.figure(figsize=(4, 6))
sns.swarmplot(
    data=GridStereo,
    y="IEEG Model Prediction",
    x="stereo EEG",
    hue="focality",
    palette="dark",
    legend=False,
    dodge=True,
)
ax = sns.boxplot(
    data=GridStereo,
    y="IEEG Model Prediction",
    x="stereo EEG",
    hue="focality",
    palette="deep",
    dodge=True,
)
plt.xlabel("")
plt.ylabel("IEEG Model")
plt.ylim(-0.04, 1.15)
h, l = ax.get_legend_handles_labels()
ax.legend(h, ["nonfocal", "focal"], loc="lower right")
plt.savefig(ospj(figpath, "seeg_ieeg.pdf"), bbox_inches="tight")

# %%
plt.figure(figsize=(4, 6))
sns.swarmplot(
    data=GridStereo,
    y="Unweighted",
    x="stereo EEG",
    hue="focality",
    palette="dark",
    legend=False,
    dodge=True,
)
ax = sns.boxplot(
    data=GridStereo,
    y="Unweighted",
    x="stereo EEG",
    hue="focality",
    palette="deep",
    dodge=True,
)
plt.ylabel("Implant Distance (mm)")
plt.ylim(0, 80)
plt.xlabel("")
h, l = ax.get_legend_handles_labels()
ax.legend(h, ["nonfocal", "focal"], loc="lower right")
plt.savefig(ospj(figpath, "seeg_distance.pdf"), bbox_inches="tight")

# %%
plt.figure(figsize=(4, 6))
sns.swarmplot(
    data=GridStereo,
    y="Preimplant Baseline Prediction",
    x="stereo EEG",
    hue="focality",
    palette="dark",
    legend=False,
    dodge=True,
)
ax = sns.boxplot(
    data=GridStereo,
    y="Preimplant Baseline Prediction",
    x="stereo EEG",
    hue="focality",
    palette="deep",
    dodge=True,
)
plt.ylabel("Implant Baseline Model")
plt.ylim(-0.04, 1.15)
plt.xlabel("")
h, l = ax.get_legend_handles_labels()
ax.legend(h, ["nonfocal", "focal"], loc="lower right")
plt.savefig(ospj(figpath, "seeg_baseline.pdf"), bbox_inches="tight")

# %%
plt.figure(figsize=(4, 6))
sns.swarmplot(
    data=GridStereo,
    y="fivesense",
    x="stereo EEG",
    hue="focality",
    palette="dark",
    legend=False,
    dodge=True,
)
ax = sns.boxplot(
    data=GridStereo,
    y="fivesense",
    x="stereo EEG",
    hue="focality",
    palette="deep",
    dodge=True,
)
plt.ylabel("5-Sense")
plt.ylim(-0.04, 1.15)
plt.xlabel("")
h, l = ax.get_legend_handles_labels()
ax.legend(h, ["nonfocal", "focal"], loc="lower right")
plt.savefig(ospj(figpath, "seeg_5s.pdf"), bbox_inches="tight")

# %%
subX = GridStereo[GridStereo["stereo EEG"] == "seeg"].drop("stereo EEG", axis=1)
subY = subX.pop("focality")

# %%
PrettyPictures(subX, subY)

# %%
subX = GridStereo[GridStereo["stereo EEG"] != "seeg"].drop("stereo EEG", axis=1)
subY = subX.pop("focality")

# %%
PrettyPictures(subX, subY)

# %%
Fig2Data = pd.concat(
    [
        subX.join(
            GoodOutcome2.map({True: "good outcome", False: "poor outcome"}).rename(
                "group"
            ),
            how="inner",
        ),
        subX.join(
            type_of_proc[type_of_proc.isin(["surgery", "device"])].rename("group"),
            how="inner",
        ),
    ],
    keys=["proc_type", "outcome"],
    names=["sep"],
).reset_index("sep")

# %% [markdown]
# #### subanalysis by type

# %%
breakdown = X.join(PreImplantData[target_col]).join(
    Base_SD.droplevel("focality"), how="inner"
)
breakdown[target_col] = (
    breakdown[target_col]
    .mask(lambda x: x == "multifocal", "broad")
    .cat.rename_categories({"broad": "broad/multifocal"})
    .cat.remove_unused_categories()
)

# %%
breakdown

# %%
test_results = []
roc_curves = []
for var, group in breakdown.melt(id_vars=target_col, ignore_index=False).groupby(
    "variable"
):
    for x, y in itertools.combinations(group.groupby(target_col), 2):
        roc = group[group[target_col].isin([x[0], y[0]])]
        roc[target_col] = roc[target_col].map({x[0]: 1, y[0]: 0})
        test_results.append(
            (
                var,
                x[0],
                y[0],
                mannwhitneyu(x[1]["value"], y[1]["value"])[1],
                real_roc(roc[target_col], roc["value"])[0],
                real_roc(roc[target_col], roc["value"])[1],
            )
        )

# %%
plt.close("all")
for SubType in ["broad/multifocal", "bifocal"]:
    plt.figure()
    plt.title(SubType)
    roc = breakdown[breakdown[target_col].isin(["unifocal", SubType])]
    roc[target_col] = roc[target_col].apply(lambda x: 1 if x == "unifocal" else 0)
    for var, group in roc.melt(id_vars=target_col, ignore_index=False).groupby(
        "variable"
    ):
        if var != "Unweighted":
            fpr, tpr, thresh = metrics.roc_curve(group[target_col], group["value"])
            plt.plot(fpr, tpr, label=var)
    plt.ylabel("Sensitivity (correctly predicting focality)")
    plt.xlabel("1-Specificity (incorrectly predicting focality)")
    # fpr, tpr,thresh = metrics.roc_curve(y_all.values, All_X['fivesense'].values)
    # plt.plot(fpr, tpr,label='Five Sense')
    plt.legend(loc="lower right")
    plt.savefig(ospj(figpath, f"{SubType[:4]}_roc.pdf"), bbox_inches="tight")

# %%
test_results = pd.DataFrame(
    test_results, columns=["variable", "group1", "group2", "p", "roc", "ci"]
)[
    lambda x: x["group1"].isin(["bifocal", "broad/multifocal", "unifocal"])
    & x["group2"].isin(["bifocal", "broad/multifocal", "unifocal"])
]
test_results

# %%
for var, data in test_results.groupby("variable"):
    print(var)
    print(data)
    print(
        sm.stats.multipletests(
            data["p"],
            method="holm",
        )
    )

# %%
plt.figure(figsize=(4, 6))
sns.swarmplot(
    data=breakdown,
    y="IEEG Model Prediction",
    x=target_col,
    palette=sns.color_palette(["#001C7F", "#08519C", "#B1400D"]),
    legend=False,
    order=["bifocal", "broad/multifocal", "unifocal"],
)
ax = sns.boxplot(
    data=breakdown,
    y="IEEG Model Prediction",
    x=target_col,
    palette=sns.color_palette(["#5B75A4", "#C0D5E2", "#CC8961"]),
    order=["bifocal", "broad/multifocal", "unifocal"],
)
plt.xlabel("ieeg determined seizure onset")
plt.ylabel("IEEG Model")
plt.ylim(-0.04, 1.15)
plt.savefig(ospj(figpath, "type_ieeg.pdf"), bbox_inches="tight")

# %%
plt.figure(figsize=(4, 6))
sns.swarmplot(
    data=breakdown,
    y="fivesense",
    x=target_col,
    palette=sns.color_palette(["#001C7F", "#08519C", "#B1400D"]),
    legend=False,
    order=["bifocal", "broad/multifocal", "unifocal"],
)
ax = sns.boxplot(
    data=breakdown,
    y="fivesense",
    x=target_col,
    palette=sns.color_palette(["#5975A4", "#A6C7D9", "#CC8963"]),
    order=["bifocal", "broad/multifocal", "unifocal"],
)
plt.ylabel("5-Sense")
plt.xlabel("ieeg determined seizure onset")
plt.ylim(-0.04, 1.15)
plt.savefig(ospj(figpath, "type_5s.pdf"), bbox_inches="tight")

# %%
plt.figure(figsize=(4, 6))
sns.swarmplot(
    data=breakdown,
    y="Preimplant Baseline Prediction",
    x=target_col,
    palette=sns.color_palette(["#001C7F", "#08519C", "#B1400D"]),
    legend=False,
    order=["bifocal", "broad/multifocal", "unifocal"],
)
ax = sns.boxplot(
    data=breakdown,
    y="Preimplant Baseline Prediction",
    x=target_col,
    palette=sns.color_palette(["#5975A4", "#A6C7D9", "#CC8963"]),
    order=["bifocal", "broad/multifocal", "unifocal"],
)
plt.ylabel("Implant Baseline Model")
plt.ylim(-0.04, 1.15)
plt.xlabel("ieeg determined seizure onset")
plt.savefig(ospj(figpath, "type_baseline.pdf"), bbox_inches="tight")

# %%
plt.figure(figsize=(4, 6))
sns.swarmplot(
    data=breakdown,
    y="Unweighted",
    x=target_col,
    palette=sns.color_palette(["#001C7F", "#08519C", "#B1400D"]),
    legend=False,
    order=["bifocal", "broad/multifocal", "unifocal"],
)
ax = sns.boxplot(
    data=breakdown,
    y="Unweighted",
    x=target_col,
    palette=sns.color_palette(["#5975A4", "#A6C7D9", "#CC8963"]),
    order=["bifocal", "broad/multifocal", "unifocal"],
)
plt.ylabel("Implant Distance (mm)")
plt.ylim(0, 80)
plt.xlabel("ieeg determined seizure onset")
plt.savefig(ospj(figpath, "type_dis.pdf"), bbox_inches="tight")

# %%
SelectPatient.index.unique().to_series().to_csv(ospj(metadata_path, "patient_list.csv"))


