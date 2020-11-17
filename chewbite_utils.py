from armetrics import utils as armutils
from armetrics import plotter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import hamming_loss, precision_recall_fscore_support

standardized_names = {"RUMIA PASTURA": "RUMIA", "PASTURA": "PASTOREO", "RUMIA PASTOREO": "RUMIA",
                      "RUMIA EN PASTURA": "RUMIA", "GRAZING": "PASTOREO", "RUMINATION": "RUMIA",
                      "R": "RUMIA", "P": "PASTOREO"}
segmentation_replacements = {"RUMIA": "SEGMENTACION", "PASTOREO": "SEGMENTACION", "REGULAR": "SEGMENTACION"}
_names_of_interest = ["PASTOREO", "RUMIA"]
_name_of_segmentation = ["SEGMENTACION"]


def load_chewbite(filename: str, start: int = None, end: int = None, verbose=True, to_segmentation=False) -> pd.Series:
    df = pd.read_table(filename, decimal=',', header=None, delim_whitespace=True, 
                       names=["bl_start", "bl_end", "label"], usecols=[0, 1, 2])

    df[["bl_start", "bl_end"]] = df[["bl_start", "bl_end"]].astype('float')

    df = df.round(0)
    df.label = df.label.str.strip().str.upper()
    df.label.replace(standardized_names, inplace=True)
    df[["bl_start", "bl_end"]] = df[["bl_start", "bl_end"]].astype('int')

    # It will modify the limits of partially selected labels
    # Given end and start may be in the middle of a label
    if start:
        df = df[df.bl_end > start]
        df.loc[df.bl_start < start, "bl_start"] = start
        df = df[df.bl_start >= start]
    if end:
        df = df[df.bl_start < end]
        df.loc[df.bl_end > end, "bl_end"] = end
        df = df[df.bl_end <= end]

    names_of_interest = _names_of_interest
    if to_segmentation:
        names_of_interest = _name_of_segmentation
        df.label.replace(segmentation_replacements, inplace=True)

    if verbose:
        print("Labels in (", start, ",", end, ") from", filename, "\n", df.label.unique())

    df = df.loc[df.label.isin(names_of_interest)]

    segments = [armutils.Segment(bl_start, bl_end, label) for name, (bl_start, bl_end, label) in df.iterrows()]
    indexes = [np.arange(bl_start, bl_end) for name, (bl_start, bl_end, label) in df.iterrows()]
    if len(segments) < 1:
        print("Warning, you are trying to load a span with no labels from:", filename)
        indexes = [np.array([])]  # To avoid errors when no blocks are present in the given interval

    frames = armutils.segments2frames(segments)
    indexes = np.concatenate(indexes)

    s = pd.Series(frames, index=indexes)

    if s.index.has_duplicates:
        print("Overlapping labels were found in", filename)
        print("Check labels corresponding to times given below (in seconds):")
        print(s.index[s.index.duplicated()])

    if len(segments) < 1:
        series_len = 1  # Provides a series with a single element that has an empty label, to avoid error
    else:
        series_len = s.index[-1]  # The series will have the length of up to the last second of the last block

    s_formatted = s.reindex(np.arange(series_len), fill_value="")

    return s_formatted


def length_signal_chewbite(filename, start=None, end=None, verbose=True, to_segmentation=False):
    df = pd.read_table(filename, decimal=',', header=None)
    df.dropna(axis=1, how='all', inplace=True)
    df.columns = ["start", "end", "label"]

    df[["start", "end"]] = df[["start", "end"]].astype('float').round(0)
    df[["start", "end"]] = df[["start", "end"]].astype('int')

    # It will modify the limits of partially selected labels
    # Given end and start may be in the middle of a label
    if start:
        df = df[df.end > start]
        df.loc[df.start < start, "start"] = start
        df = df[df.start >= start]
    if end:
        df = df[df.start < end]
        df.loc[df.end > end, "end"] = end
        df = df[df.end <= end]

    return df["end"].max() - df["start"].min()


def merge_contiguous(df):
    """ Given a dataframe df with start, end and label columns it will merge contiguous equally labeled """
    for i in df.index[:-1]:
        next_label = df.loc[i + 1].label
        if next_label == df.loc[i].label:
            df.loc[i + 1, "start"] = df.loc[i].start
            df.drop(i, inplace=True)
    return df


def remove_silences(filename_in, filename_out, max_len=300, sil_label="SILENCIO"):
    """ Given a labels filename will remove SILENCE blocks shorter than max_len (in seconds)
        if they are surrounded by blocks with the same label.
        This silences will be merged with contiguous blocks.
    """
    df = pd.read_table(filename_in, decimal=',', header=None)
    df.dropna(axis=1, how='all', inplace=True)
    df.columns = ["start", "end", "label"]

    df[["start", "end"]] = df[["start", "end"]].astype('float')
    df = df.round(0)
    df.label = df.label.str.strip().str.upper()
    df[["start", "end"]] = df[["start", "end"]].astype('int')
    
    sil_label = str.strip(str.upper(sil_label))

    for i, (start, end, label) in df.loc[df.index[1:-1]].iterrows():
        length = end - start
        prev_label = df.loc[i - 1].label
        next_label = df.loc[i + 1].label
        if label == sil_label and length <= max_len and prev_label == next_label:
            df.loc[i, "label"] = prev_label

    df = merge_contiguous(df)

    df.to_csv(filename_out,
              header=False, index=False, sep="\t")


def remove_between_given(filename_in, filename_out, search_label, max_len=300):
    """ Given a labels filename will remove blocks shorter than max_len (in seconds)
        if they are surrounded by blocks of the search_label.
        This short blocks will be merged with contiguous blocks.
    """
    df = pd.read_table(filename_in, decimal=',', header=None)
    df.dropna(axis=1, how='all', inplace=True)
    df.columns = ["start", "end", "label"]

    df[["start", "end"]] = df[["start", "end"]].astype('float')
    df = df.round(0)
    df.label = df.label.str.strip().str.upper()
    df[["start", "end"]] = df[["start", "end"]].astype('int')

    for i, (start, end, label) in df.loc[df.index[1:-1]].iterrows():
        length = end - start
        prev_label = df.loc[i - 1].label
        next_label = df.loc[i + 1].label
        if label != search_label and length <= max_len \
                and next_label == prev_label == search_label:
            df.loc[i, "label"] = search_label

    df = merge_contiguous(df)

    df.to_csv(filename_out,
              header=False, index=False, sep="\t")


def merge_file(filename_in, filename_out):
    """
    Given a labels filename_in will merge contiguous blocks and save it to filename_out.
    """
    df = pd.read_table(filename_in, decimal=',', header=None)
    df.dropna(axis=1, how='all', inplace=True)
    df.columns = ["start", "end", "label"]

    df[["start", "end"]] = df[["start", "end"]].astype('float')
    df = df.round(0)
    df.label = df.label.str.strip().str.upper()
    df[["start", "end"]] = df[["start", "end"]].astype('int')

    df = merge_contiguous(df)

    df.to_csv(filename_out,
              header=False, index=False, sep="\t")


def violinplot_metric_from_report(single_activity_report, metric):
    grouped_reports = single_activity_report.groupby("predictor_name", sort=False)
    n_predictors = len(grouped_reports)
    predictors_labels = []
    activity = single_activity_report.activity.iloc[0]

    plt.figure()
    pos = np.arange(n_predictors) + .5  # the bar centers on the y axis

    if n_predictors > 10:
        print("Be careful! I cannot plot more than 10 labels.")
    # colors = ["C%d" % i for i in range(n_predictors)]

    for (predictor_name, predictor_report), p in zip(grouped_reports, pos):
        predictors_labels.append(predictor_name)

        values_to_plot = predictor_report.loc[:, metric].values
        plt.violinplot(values_to_plot[np.isfinite(values_to_plot)], [p], points=50, vert=False, widths=0.65,
                       showmeans=False, showmedians=True, showextrema=True, bw_method='silverman')

    plt.axvline(x=0, color="k", linestyle="dashed")
    plt.axvline(x=1, color="k", linestyle="dashed")
    plt.yticks(pos, predictors_labels)
    plt.gca().invert_yaxis()
    plt.minorticks_on()
    plt.xlabel('Frame F1-score')

    plt.tight_layout()
    plt.savefig('violin_' + metric + "_" + activity + '.pdf')
    plt.savefig('violin_' + metric + "_" + activity + '.png')
    plt.show()


def my_display_report(complete_report_df):
    report_activity_grouped = complete_report_df.groupby("activity", sort=False)
    
    for activity_label, single_activity_report in report_activity_grouped:
        print("\n================", activity_label, "================\n")
        violinplot_metric_from_report(single_activity_report, "frame_f1score")


def load_chewbite2(filename: str, start: float = None, end: float = None, verbose=True, to_segmentation=False,
                   round_decimals: int = 0, frame_len: float = 1.0, names_of_interest: list = None) -> pd.DataFrame:

    blocks_in = pd.read_table(filename, decimal=',', header=None, delim_whitespace=True,
                              names=["start", "end", "label"], usecols=[0, 1, 2])

    blocks_in.loc[:, "start":"end"] = blocks_in.loc[:, "start":"end"].astype('float').round(round_decimals)
    blocks_in.label = blocks_in.label.str.strip().str.upper().replace(standardized_names)

    # It will modify the limits of partially selected labels
    # end and start may be in the middle of a label
    if start:
        blocks_in = blocks_in[blocks_in.end > start]
        blocks_in.loc[blocks_in.start < start, "start"] = start
    else:
        start = 0.0

    if end:
        blocks_in = blocks_in[blocks_in.start < end]
        blocks_in.loc[blocks_in.end > end, "end"] = end
    else:
        end = blocks_in.end.max()

    if not names_of_interest:
        names_of_interest = _names_of_interest

    if to_segmentation:
        names_of_interest = _name_of_segmentation
        blocks_in.label.replace(segmentation_replacements, inplace=True)

    if verbose:
        print("Labels in (", start, ",", end, ") from", filename, "\n", blocks_in.label.unique())

    blocks_in = blocks_in.loc[blocks_in.label.isin(names_of_interest)]
    if verbose:
        print(blocks_in)

    start_out = np.arange(start, end, frame_len)
    end_out = start_out + frame_len
    frames_out = pd.DataFrame({"start": start_out, "end": end_out, "label": ""})

    to_revise_label = "to_revise"
    for row_id, block in blocks_in.iterrows():
        criteria = (frames_out.start >= block.start) & (frames_out.end <= block.end)
        frames_out.loc[criteria, "label"] = block.label

        criteria = (frames_out.start >= block.start) & (frames_out.start < block.end) & (frames_out.end > block.end)
        frames_out.loc[criteria, "label"] = to_revise_label
        criteria = (frames_out.start < block.start) & (frames_out.end > block.start) & (frames_out.end <= block.end)
        frames_out.loc[criteria, "label"] = to_revise_label

    def revise_frame(frame):
        criteria = (blocks_in.end >= frame.start) & (blocks_in.start < frame.end)
        blocks_in_frame = blocks_in.loc[criteria].copy()

        blocks_in_frame.loc[blocks_in_frame.start < frame.start, "start"] = frame.start
        blocks_in_frame.loc[blocks_in_frame.end > frame.end, "end"] = frame.end
        blocks_in_frame["overlap"] = (blocks_in_frame.end - blocks_in_frame.start) / frame_len
        null_overlap = 1.0 - blocks_in_frame["overlap"].sum()
        blocks_in_frame = blocks_in_frame.append({"label": "", "overlap": null_overlap}, ignore_index=True)

        frame.label = blocks_in_frame.groupby("label").sum().sort_values("overlap").last_valid_index()

        return frame

    to_revise_frames = frames_out.label == to_revise_label
    frames_out.loc[to_revise_frames] = frames_out.loc[to_revise_frames].apply(revise_frame, axis="columns")

    if verbose:
        print(frames_out)

    if list(frames_out.label.unique()) == [""]:
        print("Warning, you are trying to load a span (", start, ",", end, ") with no labels from:", filename)

    return frames_out


def merge_true_pred(filename_true, filename_pred, **kwargs):
    y_true = load_chewbite2(filename_true, verbose=False, **kwargs)
    y_true.insert(0, "filename", os.path.basename(filename_true))
    y_pred = load_chewbite2(filename_pred, verbose=False, **kwargs)
    y_pred.insert(0, "filename", os.path.basename(filename_pred))
    merged = pd.merge(y_pred, y_true, on=["start", "end"],
                      how="outer", suffixes=("_pred", "_true"), sort=True).fillna("")
    return merged


def compute_cm_and_plot(df_true_pred, title=None, order=None):
    cm = pd.crosstab(df_true_pred.label_true, df_true_pred.label_pred,
                     rownames=['True'], colnames=['Predicted'], normalize="index")
    cm_abs = pd.crosstab(df_true_pred.label_true, df_true_pred.label_pred,
                     rownames=['True'], colnames=['Predicted'])
    if order:
        cm = cm.reindex(index=order, columns=order)
        cm_abs = cm_abs.reindex(index=order, columns=order)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
    sns.heatmap(cm, annot=True, cmap="Blues", cbar=None, fmt=".2f", square=True, ax=ax1)
    sns.heatmap(cm_abs, annot=True, cmap="Blues", cbar=None, fmt=".0f", square=True, ax=ax2)
    if title:
        ax1.set_title(title)
        ax2.set_title(title)
    plt.show()


def cm_single_pred(true_filenames, pred_filenames, pred_name, starts_ends=None, name_mapper=None, order=None, **kwargs):
    if starts_ends is None:
        starts_ends = [(None, None)] * len(true_filenames)

    print("\n================", pred_name, "================\n")
    to_concat = [merge_true_pred(truef, predf, start=s, end=e, **kwargs) for truef, predf, (s, e) in
                 zip(true_filenames, pred_filenames, starts_ends)]
    concatenated = pd.concat(to_concat, ignore_index=True)
    if name_mapper:
        concatenated.replace(name_mapper, inplace=True)
    compute_cm_and_plot(concatenated, title=pred_name, order=order)


def plot_predictors_cm(true_filenames, names_of_predictors, *argv_prediction_filenames, **kwargs):
    for pred_name, pred_filenames in zip(names_of_predictors, argv_prediction_filenames):
        cm_single_pred(true_filenames, pred_filenames, pred_name, **kwargs)
        

def hamming_single_pred(true_filenames, pred_filenames, pred_name, starts_ends=None, **kwargs):
    if starts_ends is None:
        starts_ends = [(None, None)] * len(true_filenames)

    to_concat = [merge_true_pred(truef, predf, start=s, end=e, **kwargs) for truef, predf, (s, e) in
                 zip(true_filenames, pred_filenames, starts_ends)]
    concatenated = pd.concat(to_concat, ignore_index=True)
     
    metrics = ["precision", "recall", "f1score", "support"]
    acts = ["RUMIA", "PASTOREO"]
        
    prfs_per_file = concatenated.groupby("filename_true").apply(
        lambda grp: precision_recall_fscore_support(grp.label_true, grp.label_pred, average=None, labels=acts))
    cols = [f"{m}_{a}" for m in metrics for a in acts]
    df_aux = pd.DataFrame(np.vstack([np.hstack(v) for v in prfs_per_file.values]), columns=cols, index=prfs_per_file.index)

    support_per_file = df_aux.iloc[:, -len(acts):].sum(axis="columns")
    support_per_file.name = "support"
    
    len_per_file = concatenated.groupby("filename_true").apply(lambda grp: len(grp.label_true))
    len_per_file.name = "length_signal"
    
    hamming_per_file = concatenated.groupby("filename_true").apply(lambda grp: hamming_loss(grp.label_true, grp.label_pred))
    
    hamming_per_file = hamming_per_file.reset_index(name="hamming_loss")
    hamming_per_file.insert(0, "predictor_name", pred_name)
    hamming_per_file = hamming_per_file.merge(support_per_file, how="left", left_on="filename_true", right_index=True)
    hamming_per_file = hamming_per_file.merge(len_per_file, how="left", left_on="filename_true", right_index=True)
    
    return hamming_per_file
        
        
def predictors_hamming_loss(true_filenames, names_of_predictors, *argv_prediction_filenames, **kwargs):
    
    to_concat = []
    
    for pred_name, pred_filenames in zip(names_of_predictors, argv_prediction_filenames):
        to_concat.append(hamming_single_pred(true_filenames, pred_filenames, pred_name, **kwargs))
        
    df_hamming = pd.concat(to_concat, ignore_index=True)
    
    df_hamming["factor_length"] = df_hamming["length_signal"] / df_hamming["length_signal"].max()
    df_hamming["hamming_length"] = df_hamming["hamming_loss"] *  df_hamming["factor_length"]

    df_hamming["factor_support"] = df_hamming["support"] / df_hamming["length_signal"]
    df_hamming["hamming_support"] = df_hamming["hamming_loss"] *  df_hamming["factor_support"]
        
    return df_hamming


def prfs_single_pred(true_filenames, pred_filenames, pred_name, starts_ends=None, average="micro", **kwargs):
    if starts_ends is None:
        starts_ends = [(None, None)] * len(true_filenames)

    to_concat = [merge_true_pred(truef, predf, start=s, end=e, **kwargs) for truef, predf, (s, e) in
                 zip(true_filenames, pred_filenames, starts_ends)]
    concatenated = pd.concat(to_concat, ignore_index=True)
    
    metrics = ["precision", "recall", "f1score", "support"]
    acts = ["RUMIA", "PASTOREO"]
        
    prfs_class = concatenated.groupby("filename_true").apply(
        lambda grp: precision_recall_fscore_support(grp.label_true, grp.label_pred, average=average, labels=acts))
    
    df_aux = pd.DataFrame(np.vstack(prfs_class.values), columns=metrics, index=prfs_class.index)
    df_aux.insert(0, "predictor_name", pred_name)
    df_aux.reset_index(inplace=True)
    
    df_aux = df_aux.astype({m: "float" for m in metrics[:-1]})

    return df_aux


def predictors_rpfs(true_filenames, names_of_predictors, *argv_prediction_filenames, **kwargs):
    
    to_concat = []
    
    for pred_name, pred_filenames in zip(names_of_predictors, argv_prediction_filenames):
        to_concat.append(prfs_single_pred(true_filenames, pred_filenames, pred_name, **kwargs))
        
    rpfs_df = pd.concat(to_concat, ignore_index=True)
        
    return rpfs_df


def f1score_single_pred(true_filenames, pred_filenames, pred_name, starts_ends=None, **kwargs):
    if starts_ends is None:
        starts_ends = [(None, None)] * len(true_filenames)

    to_concat = [merge_true_pred(truef, predf, start=s, end=e, **kwargs) for truef, predf, (s, e) in
                 zip(true_filenames, pred_filenames, starts_ends)]
    concatenated = pd.concat(to_concat, ignore_index=True)
     
    metrics = ["precision", "recall", "f1score", "support"]
    acts = ["RUMIA", "PASTOREO"]
        
    prfs_class = concatenated.groupby("filename_true").apply(
        lambda grp: precision_recall_fscore_support(grp.label_true, grp.label_pred, average=None, labels=acts))
    
    cols = [f"{m}_{a}" for m in metrics for a in acts]
    df_aux = pd.DataFrame(np.vstack([np.hstack(v) for v in prfs_class.values]), columns=cols, index=prfs_class.index)
    df_aux.insert(0, "predictor_name", pred_name)
    df_aux.reset_index(inplace=True)
    
    for p, pp in df_aux.iterrows():
        cols_sup = [(f"support_{a}", a) for a in acts]
        for c, a in cols_sup:
            if pp.loc[c] == 0:
                criteria = (concatenated.filename_true == pp.loc["filename_true"]) & (concatenated.label_pred == a)
                pred_selection = concatenated.loc[criteria]
                if pred_selection.empty:
                    df_aux.loc[pp.name, [f"{m}_{a}" for m in metrics[:3]]] = np.nan
                else:
                    print(len(pred_selection))

    return df_aux


def predictors_f1score(true_filenames, names_of_predictors, *argv_prediction_filenames, **kwargs):
    
    to_concat = []
    
    for pred_name, pred_filenames in zip(names_of_predictors, argv_prediction_filenames):
        to_concat.append(f1score_single_pred(true_filenames, pred_filenames, pred_name, **kwargs))
        
    return pd.concat(to_concat, ignore_index=True)