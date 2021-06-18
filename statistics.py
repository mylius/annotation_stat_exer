import json
import numpy as np
import pandas as pd
from os.path import isfile
import plotly.express as px
from sklearn.metrics import classification_report
import argparse
from warnings import warn
from sklearn.preprocessing import normalize

default_font_size = 15


def annodata_to_df():
    """
    This function reads the annotation data and the reference data and creates a annotator focused multilevel pandas dataframe.
    """
    with open("./data/anonymized_project.json", "r") as f:
        data = json.load(f)
    results = data["results"]["root_node"]["results"]
    sessions = list(results.keys())
    reference = pd.read_json(r"./data/references.json").T
    annotators = {}
    # df = pd.read_json(r"./data/references.json").T
    colnames = ["answer", "cant_solve", "corrupt_data", "duration"]
    columns = []
    # create a dictionary with all annoators and their annotations results.
    for session in sessions:
        for result in results[session]["results"]:
            try:
                annotators[result["user"]["vendor_user_id"]].append(result)
            except:
                annotators[result["user"]["vendor_user_id"]] = []
                annotators[result["user"]["vendor_user_id"]].append(result)

    # creates the columns for the DataFrane
    for annotator, data in annotators.items():
        for col in colnames:
            columns.append((annotator, col))
    reference = pd.read_json(r"./data/references.json").T
    micolumns = pd.MultiIndex.from_tuples(columns)

    df = pd.DataFrame(columns=micolumns, index=reference.index)
    # populate the DataFrame with the data.
    """I didn't find any elegant way to create the dataframe and populate it at the same time,
     because I needed the list of annotators to create the columns. 
    Creating them one after another turned out to be a bit messy. 
    This would be an improvement for the future."""
    for annotator, data in annotators.items():
        for item in data:
            image_id = item["task_input"]["image_url"][-12:-4]
            df.loc[image_id, (annotator, "answer")] = item["task_output"]["answer"]
            df.loc[image_id, (annotator, "cant_solve")] = item["task_output"][
                "cant_solve"
            ]
            df.loc[image_id, (annotator, "corrupt_data")] = item["task_output"][
                "corrupt_data"
            ]
            df.loc[image_id, (annotator, "duration")] = item["task_output"][
                "duration_ms"
            ]
    return df.sort_index().sort_index(axis=1)


def get_duration_stat(df, plot=True):
    """
    Takes the annotations DataFrame and returns a Dataframe containing mean,min and max annotations times for each annotator.
    If plot is True a plot of the data will be created.

    Arguments
    ----------
    df: The annotations DataFrame created by annodata_to_df()
    plot: A Boolean
    """
    mean_durations = (
        df.xs("duration", level=1, axis=1, drop_level=False).mean(axis=0).droplevel(1)
    )
    min_durations = (
        df.xs("duration", level=1, axis=1, drop_level=False).min(axis=0).droplevel(1)
    )
    max_durations = (
        df.xs("duration", level=1, axis=1, drop_level=False).max(axis=0).droplevel(1)
    )
    durations_df = pd.concat(
        [mean_durations, min_durations, max_durations], axis=1, levels=1
    ).sort_index()
    durations_df.columns = ["mean", "min", "max"]
    durations_df.index = durations_df.index.set_names(["annotators"])
    dur_smaller_zero = min_durations[min_durations < 0]
    if len(dur_smaller_zero) == 1:
        warn("Warning: {} has durations < 0.".format(list(dur_smaller_zero.index)))
    if len(dur_smaller_zero) > 1:
        warn("Warning: {} have durations < 0.".format(list(dur_smaller_zero.index)))
    idx_temp = durations_df.index
    if plot:
        durations_df.index = range(1, 23)
        fig_dur = px.bar(
            durations_df,
            y=["min", "max", "mean"],
            barmode="group",
            # title="Annotations Times",
        )
        fig_dur.update_layout(
            {"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)",}
        )
        fig_dur.update_yaxes(title_text="Time[s]")
        fig_dur.update_layout(
            xaxis=dict(dtick=1, tickmode="linear", title_text="Annotator"),
            font=dict(size=default_font_size),
        )
        fig_dur.show()
    durations_df.index = idx_temp
    return durations_df


def get_num_of_annot_stat(df, plot=True):
    """
    Takes the annotations DataFrame and returns a Dataframe containing number of Annotations for each annotator.
    If plot is True a plot of the data will be created.

    Arguments
    ----------
    df: The annotations DataFrame created by annodata_to_df()
    plot: A Boolean
    """
    num_df = pd.DataFrame(
        df.xs("duration", level=1, axis=1, drop_level=False).count().droplevel(1),
        columns=["#Annotations"],
    ).sort_index()
    num_df.index = num_df.index.set_names(["annotators"])
    idx_temp = num_df.index
    if plot:
        num_df.index = range(1, 23)
        fig_num_anno = px.bar(
            num_df,
            y=["#Annotations"],
            barmode="group",  # title="Number of Annotations"
        )
        fig_num_anno.update_layout(
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            font=dict(size=default_font_size),
        )
        fig_num_anno.update_traces(textposition="inside", textfont_size=14)
        fig_num_anno.update_layout(
            xaxis=dict(dtick=1, tickmode="linear", title_text="Annotator")
        )
        fig_num_anno.show()
    num_df.index = idx_temp
    return num_df


def get_disagreement(df, thresh=0.4):
    """
    Takes the annotations DataFrame and a threshold and returns a Dataframe containing the questions annotators disagreed on the most.
    Cut off is determined by the threshold.

    Arguments
    ----------
    df: The annotations DataFrame created by annodata_to_df()
    thresh: A float
    """
    answer_df = df.xs("answer", level=1, axis=1, drop_level=False).droplevel(1, axis=1)
    temp = answer_df.T.apply(pd.Series.value_counts, axis=0).fillna(0)
    temp = temp.drop([""])
    no_smaller = temp.T[temp.T["no"] < temp.T["yes"]]
    yes_smaller = temp.T[temp.T["no"] > temp.T["yes"]]
    agreement_df = abs(
        1
        - pd.concat(
            [
                yes_smaller["yes"] / yes_smaller["no"],
                no_smaller["no"] / no_smaller["yes"],
            ]
        )
        / 0.5
    ).sort_index()
    disagreable = agreement_df[agreement_df < thresh]
    return disagreable


def plot_ref_balance(plot=True):
    """
    Returns a DataFrame containing the number of true and false values in the reference Dataset.
    If plot is True a pie chart of the balance will be plotted.

    Arguments
    ----------
    plot: A Boolean
    """
    reference = pd.read_json(r"./data/references.json").T
    if plot:
        fig_balance = px.pie(
            reference, names="is_bicycle",  # title="Balance of the reference set"
        )
        fig_balance.update_layout(
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            font=dict(size=18),
        )
        fig_balance.show()
    return reference["is_bicycle"].value_counts()


def calc_f1(df, plot=True):
    """
    Takes the annotations DataFrame and returns a Dataframe containing a classification report for each annotator.
    If plot is the f1 score will be plotted.

    Arguments
    ----------
    df: The annotations DataFrame created by annodata_to_df()
    plot: A Boolean
    """
    reference = pd.read_json(r"./data/references.json").T
    answer_df = df.xs("answer", level=1, axis=1, drop_level=False).droplevel(1, axis=1)
    classification = {}
    for annotator in answer_df.columns:
        ref_array = (
            reference["is_bicycle"][answer_df[annotator].dropna().index]
            .sort_index()
            .values
        )
        annot_array = (
            answer_df[annotator]
            .dropna()
            .sort_index()
            .map(dict(yes=True, no=False))
            .values
        )
        # Weirdly annot_array sometimes contains bools instead of np.bool_s
        if type(annot_array[0]) == bool:
            annot_array = [np.bool_(i) for i in annot_array]
        classification[annotator] = pd.DataFrame(
            classification_report(ref_array, annot_array, output_dict=True)
        )
    class_df = pd.concat(classification, axis=1)
    if plot:
        class_df_plot = (
            class_df.xs("accuracy", level=1, axis=1, drop_level=False)
            .iloc[0]
            .T.droplevel(1)
        )
        class_df_plot.index = range(1, 23)
        fig_dur = px.bar(class_df_plot)
        fig_dur.update_layout(
            {"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)",}
        )
        fig_dur.update_yaxes(title_text="f1-score")
        fig_dur.update_layout(
            xaxis=dict(dtick=1, tickmode="linear", title_text="Annotator"),
            font=dict(size=default_font_size),
        )
        fig_dur.show()
    return class_df


def get_unsolved(df, plot=True):
    """
    Takes the annotations DataFrame and returns a Dataframe containing the percentage of can't solve answers for each annotator.
    If plot is True a plot of the data will be created.

    Arguments
    ----------
    df: The annotations DataFrame created by annodata_to_df()
    plot: A Boolean
    """
    # cant_solve
    cant_solve = df.xs("cant_solve", level=1, axis=1, drop_level=False).droplevel(
        1, axis=1
    )
    cant_solve_dict = {}
    for annotator in df.xs("duration", level=1, axis=1, drop_level=False).droplevel(
        1, axis=1
    ):
        cant_solve_dict[annotator] = cant_solve[annotator].value_counts()
    cant_solve = pd.concat(cant_solve_dict, axis=1)
    print("Number of times an image was labeld as unsolvable:")
    print(cant_solve.loc[True].sum())
    cant_solve = pd.DataFrame(
        cant_solve.fillna(0).iloc[1] * 100 / cant_solve.iloc[0],
        columns=['% "can\'t solves"'],
    )
    # corrupt
    corrupt_annot = df.xs("corrupt_data", level=1, axis=1, drop_level=False).droplevel(
        1, axis=1
    )
    corrupt_annot_dict = {}
    for annotator in df.xs("duration", level=1, axis=1, drop_level=False).droplevel(
        1, axis=1
    ):
        corrupt_annot_dict[annotator] = corrupt_annot[annotator].value_counts()
    corrupt_annot = pd.concat(corrupt_annot_dict, axis=1)
    print("Number of times an image was labeld as corrupt:")
    print(corrupt_annot.loc[True].sum())
    corrupt_annot = pd.DataFrame(
        corrupt_annot.fillna(0).iloc[1] * 100 / corrupt_annot.iloc[0],
        columns=['% "corrupt"'],
    )
    unsolved_df = pd.concat([corrupt_annot, cant_solve], axis=1)
    idx_temp = unsolved_df.index
    if plot:
        unsolved_df.index = range(1, 23)
        fig_unsol = px.bar(
            unsolved_df,
            y=['% "can\'t solves"', '% "corrupt"'],
            barmode="group",
            # title="Percentage of Unsolved Questions",
        )
        fig_unsol.update_layout(
            {"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)",}
        )
        fig_unsol.update_yaxes(title_text="%")
        fig_unsol.update_layout(
            xaxis=dict(dtick=1, tickmode="linear", title_text="Annotator"),
            font=dict(size=default_font_size),
        )
        fig_unsol.show()
    print("Percentage of Unsolved Questions:")
    unsolved_df.index = idx_temp
    return unsolved_df


def scale(X, x_min, x_max):
    """Taken from https://datascience.stackexchange.com/questions/39142/normalize-matrix-in-python-numpy"""
    nom = (X - X.min(axis=0)) * (x_max - x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom = denom + (denom is 0)
    return x_min + nom / denom


def annotator_quality(df,duration,unsolved, weights=[2, 0.5, 0.5], plot=True):
    """
    Takes the annotations DataFrame and returns a Dataframe a metric for the quality of each annotator.
    The metric takes three things into account: f1-score, number of unsolved questions, and mean time.
    A weight can be applied to each of these by passing a list of weights.
    If plot is True a plot of the data will be created.

    Arguments
    ----------
    df: The annotations DataFrame created by annodata_to_df()
    weights: A list of three floats
    plot: A Boolean
    """
    class_df = calc_f1(df, plot)
    print(class_df)
    f1_df = (
        class_df.xs("accuracy", level=1, axis=1, drop_level=False)
        .iloc[0]
        .T.droplevel(1)
    )
    accuracy = scale(f1_df, 0, 1)
    unsolved = unsolved
    cant = scale(unsolved['% "can\'t solves"'], 0, 1)
    mean_time = scale(duration["mean"], 0, 1)
    quality = weights[0] * accuracy - weights[1] * cant - weights[2] * mean_time
    idx_temp = quality.index
    if plot:
        quality.index = range(1, 23)
        fig_qual = px.bar(quality)
        fig_qual.update_layout(
            {"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)",}
        )
        fig_qual.update_yaxes(title_text="Score")
        fig_qual.update_layout(
            xaxis=dict(dtick=1, tickmode="linear", title_text="Annotator"),
            font=dict(size=default_font_size),
        )
        fig_qual.show()
    quality.index = idx_temp
    print("\nQuality of Annotators:")
    return quality


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistics on Annotation Data")
    parser.add_argument(
        "-p", action="store_true", help="If set the data will be plotted.",
    )
    parser.add_argument(
        "-r", action="store_true", help="If set the data will be reinitialized.",
    )
    plot = parser.parse_args().p
    reinit = parser.parse_args().r
    try:
        assert not reinit
        df = pd.read_pickle("./.dataframe.pkl")
    except:
        df = annodata_to_df()
        df.to_pickle("./.dataframe.pkl")
    # get_corrupt_annot(df)

    num_annotators = (
        df.xs("duration", level=1, axis=1, drop_level=False)
        .droplevel(1, axis=1)
        .shape[1]
    )
    # 1a

    print("Total Number of Annotators:")
    print(num_annotators)
    # 1b
    print("\nAnswer Durations:")
    durations = get_duration_stat(df, plot)
    print(durations)
    # 1c
    print("\nNumber of Annotations:")
    print(get_num_of_annot_stat(df, plot))
    # 1d
    print("\nQuestions Annotaters disagreed on:")
    print(get_disagreement(df))
    # 2a
    print("\nUnsolved Questions:")
    unsolved = get_unsolved(df, plot)
    print(unsolved)
    # 3
    print("\nReference Balance:")
    print(plot_ref_balance(plot))
    # 4
    print("\nMatch Between Reference and Annotations:")
    print(annotator_quality(df,durations, unsolved,plot=plot))
