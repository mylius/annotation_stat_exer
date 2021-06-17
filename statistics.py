import json
import numpy as np
import pandas as pd
from os.path import isfile
import plotly.express as px
from sklearn.metrics import classification_report

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
    return df.sort_index()


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
    if plot:
        fig_dur = px.bar(
            durations_df,
            y=["min", "max", "mean"],
            barmode="group",
            title="Annotations Times",
        )
        fig_dur.update_yaxes(title_text="Time[s]")
        fig_dur.show()
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
        df.xs("answer", level=1, axis=1, drop_level=False).count().droplevel(1),
        columns=["#Annotations"],
    ).sort_index()
    num_df.index = num_df.index.set_names(["annotators"])
    if plot:
        fig_num_anno = px.bar(
            num_df, y=["#Annotations"], barmode="group", title="Number of Annotations"
        )
        fig_num_anno.show()
    return num_df


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
            reference, names="is_bicycle", title="Balance of the reference set"
        )
        fig_balance.show()
    return reference["is_bicycle"].value_counts()

def calc_f1(df,plot = True):
    """
    Takes the annotations DataFrame and returns a Dataframe containing a classification report for each annotator.
    If plot is the f1 score will be plotted.

    Arguments
    ----------
    df: The annotations DataFrame created by annodata_to_df()
    plot: A Boolean
    """
    reference = pd.read_json(r"./data/references.json").T
    answer_df = df.xs("answer", level=1, axis=1, drop_level=False).droplevel(1,axis = 1)
    classification = {}
    for annotator in answer_df.columns:
        ref_array = reference["is_bicycle"][answer_df[annotator].dropna().index].sort_index().values
        annot_array = answer_df[annotator].dropna().sort_index().map(dict(yes=True, no=False)).values
        #Weirdly annot_array sometimes contains bools instead of np.bool_s
        if type(annot_array[0]) == bool:
            annot_array = [np.bool_(i) for i in annot_array]
        classification[annotator] = pd.DataFrame(classification_report(ref_array,annot_array, output_dict=True))
    class_df = pd.concat(classification,axis = 1)
    if plot:
        class_df_plot = class_df.xs("accuracy", level=1, axis=1, drop_level=False).iloc[0].T.droplevel(1)
        fig_dur = px.bar(
            class_df_plot,
            y=["f1-score"],
            barmode="group",
            title="F1-Score of the Accuracy",
        )
        fig_dur.update_yaxes(title_text="Time[s]")
        fig_dur.show()
    return class_df

if __name__ == "__main__":
    try:
        df = pd.read_pickle("./.dataframe.pkl")
    except:
        df = annodata_to_df()
        df.to_pickle("./.dataframe.pkl")
    num_annotators = df.xs("answer", level=1, axis=1, drop_level=False).droplevel(1,axis=1).shape[1]
    num_annotations = {}
    get_duration_stat(df,plot=False)
    get_num_of_annot_stat(df, plot=False)
    plot_ref_balance(plot=False)
    print(calc_f1(df,plot=False))
