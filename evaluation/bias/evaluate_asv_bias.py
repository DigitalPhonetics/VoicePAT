import bt4vt
import os
import pandas as pd
import numpy as np

def clean_scores_file(filename):

    header_names = ["ref_file", "com_file", "sc"]
    scores_file = pd.read_csv(filename, names=header_names, sep=" ")

    # remove "F" or "M"
    scores_file["ref_file"] = scores_file["ref_file"].str[1:]
    scores_file["com_file"] = scores_file["com_file"].str[1:]

    # add label column
    scores_file["lab"] = np.where((scores_file["ref_file"] == scores_file["com_file"].str[:4]), 1, 0)

    return scores_file

def evaluate_asv_bias(scores_file, bt4vt_config_file):

    scores_data = clean_scores_file(scores_file)
    test = bt4vt.core.SpeakerBiasTest(scores_data, str(bt4vt_config_file))
    test.run_tests()
    
    return test