# Ref: https://www.kaggle.com/code/raddar/amex-data-int-types-train
import numpy as np
import pandas as pd


def floorify_frac(x, interval: [int, float]):
    """convert to int if float appears ordinal"""
    xt = (np.floor(x / interval + 1e-6)).fillna(-1)
    if np.max(xt) <= 127:
        return xt.astype(np.int8)
    return xt.astype(np.int16)


def floorify_random_noise_b(x: pd.DataFrame):
    x["B_4"] = floorify_frac(x["B_4"], 1 / 78)
    x["B_16"] = floorify_frac(x["B_16"], 1 / 12)
    x["B_20"] = floorify_frac(x["B_20"], 1 / 17)
    x["B_22"] = floorify_frac(x["B_22"], 1 / 2)
    x["B_30"] = floorify_frac(x["B_30"], 1)
    x["B_31"] = floorify_frac(x["B_31"], 1)
    x["B_32"] = floorify_frac(x["B_32"], 1)
    x["B_33"] = floorify_frac(x["B_33"], 1)
    x["B_38"] = floorify_frac(x["B_38"], 1)
    x["B_41"] = floorify_frac(x["B_41"], 1)

    return x


def floorify_random_noise_d(x: pd.DataFrame):
    x["D_39"] = floorify_frac(x["D_39"], 1 / 34)
    x["D_44"] = floorify_frac(x["D_44"], 1 / 8)
    x["D_49"] = floorify_frac(x["D_49"], 1 / 71)
    x["D_51"] = floorify_frac(x["D_51"], 1 / 3)
    x["D_59"] = floorify_frac(x["D_59"] + 5 / 48, 1 / 48)
    x["D_65"] = floorify_frac(x["D_65"], 1 / 38)
    x["D_66"] = floorify_frac(x["D_66"], 1)
    x["D_68"] = floorify_frac(x["D_68"], 1)
    x["D_70"] = floorify_frac(x["D_70"], 1 / 4)
    x["D_72"] = floorify_frac(x["D_72"], 1 / 3)
    x["D_74"] = floorify_frac(x["D_74"], 1 / 14)
    x["D_75"] = floorify_frac(x["D_75"], 1 / 15)
    x["D_78"] = floorify_frac(x["D_78"], 1 / 2)
    x["D_79"] = floorify_frac(x["D_79"], 1 / 2)
    x["D_80"] = floorify_frac(x["D_80"], 1 / 5)
    x["D_81"] = floorify_frac(x["D_81"], 1)
    x["D_82"] = floorify_frac(x["D_82"], 1 / 2)
    x["D_83"] = floorify_frac(x["D_83"], 1)
    x["D_84"] = floorify_frac(x["D_84"], 1 / 2)
    x["D_86"] = floorify_frac(x["D_86"], 1)
    x["D_87"] = floorify_frac(x["D_87"], 1)
    x["D_89"] = floorify_frac(x["D_89"], 1 / 9)
    x["D_91"] = floorify_frac(x["D_91"], 1 / 2)
    x["D_92"] = floorify_frac(x["D_92"], 1)
    x["D_93"] = floorify_frac(x["D_93"], 1)
    x["D_94"] = floorify_frac(x["D_94"], 1)
    x["D_96"] = floorify_frac(x["D_96"], 1)
    x["D_103"] = floorify_frac(x["D_103"], 1)
    x["D_106"] = floorify_frac(x["D_106"], 1 / 23)
    x["D_107"] = floorify_frac(x["D_107"], 1 / 3)
    x["D_108"] = floorify_frac(x["D_108"], 1)
    x["D_109"] = floorify_frac(x["D_109"], 1)
    x["D_111"] = floorify_frac(x["D_111"], 1 / 2)
    x["D_113"] = floorify_frac(x["D_113"], 1 / 5)
    x["D_114"] = floorify_frac(x["D_114"], 1)
    x["D_116"] = floorify_frac(x["D_116"], 1)
    x["D_117"] = floorify_frac(x["D_117"] + 1, 1)
    x["D_120"] = floorify_frac(x["D_120"], 1)
    x["D_122"] = floorify_frac(x["D_122"], 1 / 7)
    x["D_123"] = floorify_frac(x["D_123"], 1)
    x["D_124"] = floorify_frac(x["D_124"] + 1 / 22, 1 / 22)
    x["D_125"] = floorify_frac(x["D_125"], 1)
    x["D_126"] = floorify_frac(x["D_126"] + 1, 1)
    x["D_127"] = floorify_frac(x["D_127"], 1)
    x["D_129"] = floorify_frac(x["D_129"], 1)
    x["D_135"] = floorify_frac(x["D_135"], 1)
    x["D_136"] = floorify_frac(x["D_136"], 1 / 4)
    x["D_137"] = floorify_frac(x["D_137"], 1)
    x["D_138"] = floorify_frac(x["D_138"], 1 / 2)
    x["D_139"] = floorify_frac(x["D_139"], 1)
    x["D_140"] = floorify_frac(x["D_140"], 1)
    x["D_143"] = floorify_frac(x["D_143"], 1)
    x["D_145"] = floorify_frac(x["D_145"], 1 / 11)

    x["D_63"] = (
        x["D_63"]
        .apply(lambda t: {"CR": 1, "XZ": 2, "XM": 3, "CO": 4, "CL": 5, "XL": 6}[t])
        .astype(np.int8)
    )
    x["D_64"] = x["D_64"].fillna("NULL")
    x["D_64"] = (
        x["D_64"]
        .apply(lambda t: {"NULL": 1, "O": 2, "-1": 3, "R": 4, "U": 5}[t])
        .astype(np.int8)
    )

    return x


def floorify_random_noise_r(x: pd.DataFrame):
    x["R_2"] = floorify_frac(x["R_2"], 1)
    x["R_3"] = floorify_frac(x["R_3"], 1 / 10)
    x["R_4"] = floorify_frac(x["R_4"], 1)
    x["R_5"] = floorify_frac(x["R_5"], 1 / 2)
    x["R_8"] = floorify_frac(x["R_8"], 1)
    x["R_9"] = floorify_frac(x["R_9"], 1 / 6)
    x["R_10"] = floorify_frac(x["R_10"], 1)
    x["R_11"] = floorify_frac(x["R_11"], 1 / 2)
    x["R_13"] = floorify_frac(x["R_13"], 1 / 31)
    x["R_15"] = floorify_frac(x["R_15"], 1)
    x["R_16"] = floorify_frac(x["R_16"], 1 / 2)
    x["R_17"] = floorify_frac(x["R_17"], 1 / 35)
    x["R_18"] = floorify_frac(x["R_18"], 1 / 31)
    x["R_19"] = floorify_frac(x["R_19"], 1)
    x["R_20"] = floorify_frac(x["R_20"], 1)
    x["R_21"] = floorify_frac(x["R_21"], 1)
    x["R_22"] = floorify_frac(x["R_22"], 1)
    x["R_23"] = floorify_frac(x["R_23"], 1)
    x["R_24"] = floorify_frac(x["R_24"], 1)
    x["R_25"] = floorify_frac(x["R_25"], 1)
    x["R_26"] = floorify_frac(x["R_26"], 1 / 28)
    x["R_28"] = floorify_frac(x["R_28"], 1)

    return x


def floorify_random_noise_s(x: pd.DataFrame):
    x["S_6"] = floorify_frac(x["S_6"], 1)
    x["S_11"] = floorify_frac(x["S_11"] + 5 / 25, 1 / 25)
    x["S_15"] = floorify_frac(x["S_15"] + 3 / 10, 1 / 10)
    x["S_18"] = floorify_frac(x["S_18"], 1)
    x["S_20"] = floorify_frac(x["S_20"], 1)

    return x
