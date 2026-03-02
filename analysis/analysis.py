import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    # Data Prep
    return


@app.cell
def _():
    import marimo as mo 
    import pandas as pd 
    import pyarrow as pa 
    from pathlib import Path 
    import numpy as np
    from great_tables import GT, md, html
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    import seaborn as sns 
    from scipy.stats import friedmanchisquare, wilcoxon
    from itertools import combinations
    from statsmodels.stats.multitest import multipletests
    from sigfig import round as sig_round
    import re

    question_data = Path('dataset') / 'PCOS_Guideline_Dataset_srtype.xlsm'
    original_fullset_path = Path('dataset') / 'fullgroundtruth_valid_apimerge_df.parquet'
    evalset_path = Path('dataset') / 'groundtruth_eval.parquet'
    metrics_path = Path('dataset') / 'overall_evalmetrics_df_analysis.xlsx'

    original_retrieved_df = pd.read_parquet(original_fullset_path)
    original_retrieved_df['standardised_question_id'] = original_retrieved_df['question_id'].str.replace('/', ',')
    evalset_df = pd.read_parquet(evalset_path)
    evalset_df['standardised_question_id'] = evalset_df['question_id'].str.replace('/', ',')
    # Define the desired order of databases
    database_order = ['embase', 'medline', 'pubmed', 'openalex']  # Adjust this list to match your exact database names

    # Convert the 'database' column to a categorical type with the specified order

    overarching_novs_results_df = pd.read_excel(metrics_path, sheet_name = 'ovearching_no_vs')
    overarching_vs_results_df = pd.read_excel(metrics_path, sheet_name = 'overarching_vs')
    topicspecific_novs_results_df = pd.read_excel(metrics_path, sheet_name = 'topic_specific_no_vs')
    topicspecific_vs_results_df = pd.read_excel(metrics_path, sheet_name = 'topic-specific_vs')
    topicspecific_vs_results_df = topicspecific_vs_results_df.fillna(0)
    topicspecific_vs_results_df['database'] = pd.Categorical(
        topicspecific_vs_results_df['database'], 
        categories=database_order, 
        ordered=True
    )
    topicspecific_vs_results_df = topicspecific_vs_results_df.sort_values('database')
    topicspecific_vs_results_df.rename(columns = {'evidence_review_id' : 'question_id'}, inplace=True)
    overarching_vs_results_df

    #standardize question ids 
    topicspecific_vs_results_df['standardised_question_id'] = topicspecific_vs_results_df['question_id'].str.replace('/', ',')
    topicspecific_novs_results_df['standardised_question_id'] =topicspecific_novs_results_df['question_id'].str.replace('/', ',')

    #question ids not in evalset 
    rq_data = pd.read_excel(question_data, sheet_name = 'rq_evidence_review')
    rq_data['Item'] = rq_data['Item'].astype(str)
    rq_data['standardised_question_id'] = (rq_data['Item']
                                          .astype(str)
                                          .str.replace('/', ',')  # Replace slashes with commas
                                          .str.strip()  # Remove leading/trailing whitespace
                                          .str.replace(r'\s*,\s*', ',', regex=True)  # Remove spaces around commas
                                          .str.replace(r'\.$', '', regex=True))  # 
    excluded_questions = set(rq_data['standardised_question_id']) - set(evalset_df['standardised_question_id']) 


    # Check if all elements in the first set are in the second set
    assert set(topicspecific_vs_results_df['standardised_question_id']).issubset(set(rq_data['standardised_question_id']))

    gdg_theme_mapping = dict(zip(evalset_df['standardised_question_id'], evalset_df['GDG'].astype('str')))
    sr_update_mapping = dict(zip(evalset_df['standardised_question_id'], evalset_df['sr_update']))
    sr_type_mapping = dict(zip(rq_data['standardised_question_id'], rq_data['sr_type']))
    topicspecific_vs_results_df['gdg'] = topicspecific_vs_results_df['standardised_question_id'].map(gdg_theme_mapping)
    topicspecific_vs_results_df['sr_update'] = topicspecific_vs_results_df['standardised_question_id'].map(sr_update_mapping)
    topicspecific_vs_results_df['sr_type'] = topicspecific_vs_results_df['standardised_question_id'].map(sr_type_mapping)
    topicspecific_novs_results_df['gdg'] = topicspecific_novs_results_df['standardised_question_id'].map(gdg_theme_mapping)
    topicspecific_novs_results_df['sr_update'] = topicspecific_novs_results_df['standardised_question_id'].map(sr_update_mapping)
    topicspecific_novs_results_df['sr_type'] = topicspecific_novs_results_df['standardised_question_id'].map(sr_type_mapping)
    return (
        GT,
        evalset_df,
        mo,
        np,
        overarching_novs_results_df,
        overarching_vs_results_df,
        pd,
        plt,
        sig_round,
        sns,
        stats,
        topicspecific_novs_results_df,
        topicspecific_vs_results_df,
    )


@app.cell
def _(evalset_df):
    evalset_df
    return


@app.cell
def _(topicspecific_novs_results_df):
    def iqr(x): 
        return x.quantile(0.75) - x.quantile(0.25)

    _calc_medianrelevant = topicspecific_novs_results_df.query("database == 'embase'").copy() 
    _calc_medianrelevant['total_relevant'] = _calc_medianrelevant['n_retrieved'] + _calc_medianrelevant['n_missed']

    medianrelevant = _calc_medianrelevant['total_relevant'].median()
    iqr_relevant =  iqr(_calc_medianrelevant['total_relevant']) 

    print(f"Median Total Relevant (Embase): {medianrelevant}")
    print(f"IQR Total Relevant (Embase): {iqr_relevant}")
    return (iqr,)


@app.cell
def _(mo):
    mo.md(r"""
    # Global Level Performance
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Calculate global performance for topic specific vector searches
    """)
    return


@app.cell
def _(sig_round, topicspecific_vs_results_df):


    metric_col = ['recall', 'precision', 'f1_score', 'f2_score', 'f3_score', 'overall_screening_workload']
    adjusted_metric_col = [col+'_adjusted' for col in metric_col]
    cutoff_col = [col + '_cutoff' for col in metric_col]
    adjusted_cutoff_col = [col+'_adjusted' for col in cutoff_col]

    col_interest_vs = [metric_col + adjusted_metric_col + cutoff_col + adjusted_cutoff_col]
    grouping_col = ['database', 'vector_search_type']

    raw_metrics = ['n_retrieved', 'n_missed', 'overall_screening_workload', 'n_retrieved_adjusted', 'n_missed_adjusted', 'overall_screening_workload_cutoff'] + grouping_col

    topicspecific_vs_global = topicspecific_vs_results_df[raw_metrics].groupby(grouping_col).sum().reset_index()
    # Calculate metrics for both regular and adjusted columns
    def calc_metrics(df, metrics):
        for retrieved, missed, workload, suffix in metrics:

            # Calculate recall
            df[f'recall{suffix}'] = df.apply(
                lambda row: sig_round(row[retrieved] / (row[retrieved] + row[missed]), sigfigs=4),
                axis=1
            )

            # Calculate precision
            df[f'precision{suffix}'] = df.apply(
                lambda row: sig_round(row[retrieved] / row[workload], sigfigs=4), 
                axis=1
            )

            # Calculate NNR (Number Needed to Read)
            df[f'nnr{suffix}'] = df.apply(
                lambda row: sig_round(1 / row[f'precision{suffix}'], sigfigs=4) if row[f'precision{suffix}'] != 0 else float('inf'),
                axis=1
            )

            # Calculate F-beta scores for beta values 1-3
            for beta in range(1, 4):
                df[f'f{beta}_score{suffix}'] = df.apply(
                    lambda row: sig_round(
                        (1 + beta**2) * row[f'precision{suffix}'] * row[f'recall{suffix}'] / 
                        (beta**2 * row[f'precision{suffix}'] + row[f'recall{suffix}']),
                        sigfigs=4
                    ) if (beta**2 * row[f'precision{suffix}'] + row[f'recall{suffix}']) > 0 else 0,
                    axis=1
                )
        return df
    # Define base and adjusted metrics

    metrics = [
        # (base_retrieved, base_missed, base_workload, suffix)
        ('n_retrieved', 'n_missed', 'overall_screening_workload_cutoff', ''),
        ('n_retrieved_adjusted', 'n_missed_adjusted', 'overall_screening_workload_cutoff', '_adjusted')
    ]

    topicspecific_vs_global = calc_metrics(topicspecific_vs_global, metrics)
    topicspecific_vs_global.head()
    return (
        adjusted_cutoff_col,
        adjusted_metric_col,
        cutoff_col,
        metric_col,
        topicspecific_vs_global,
    )


@app.cell
def _():
    ## Prep Global Level Results Dataframe
    return


@app.cell
def _(
    adjusted_cutoff_col,
    adjusted_metric_col,
    cutoff_col,
    metric_col,
    overarching_novs_results_df,
    overarching_vs_results_df,
    pd,
    topicspecific_novs_results_df,
    topicspecific_vs_global,
):
    topicspecific_novs_results_df['Workflow'] = 'Workflow 1 (T)'
    overarching_novs_results_df['Workflow'] = 'Workflow 2 (O)'
    overarching_vs_results_df['Workflow'] = 'Workflow 3 (O + OVS)'
    topicspecific_vs_global['Workflow'] = 'Workflow 4 (O + TVS)'
    topicspecific_novs_global = topicspecific_novs_results_df.query('question_id=="overall"').copy()


    overarching_vs = overarching_vs_results_df.copy()
    overarching_vs.drop(columns = metric_col, inplace=True)
    overarching_vs[metric_col]= overarching_vs_results_df[cutoff_col]
    overarching_vs.drop(columns = cutoff_col, inplace=True)
    overarching_vs.drop(columns = adjusted_metric_col, inplace = True)
    overarching_vs[adjusted_metric_col] = overarching_vs_results_df[adjusted_cutoff_col]
    overarching_vs.drop(columns = adjusted_cutoff_col, inplace=True)


    topicspecific_vs = topicspecific_vs_global.drop(columns = 'overall_screening_workload').copy()
    topicspecific_vs['overall_screening_workload'] = topicspecific_vs_global['overall_screening_workload_cutoff']


    overall_performance_df = pd.concat([overarching_novs_results_df,overarching_vs, topicspecific_novs_global,topicspecific_vs])

    overall_performance_df.drop(columns = ['search_type', 'strategy_type', 'question_id', 'performance_on'], inplace=True)
    overall_performance_df['vector_search'] = overall_performance_df['vector_search'].fillna(True)
    overall_performance_df['nnr'] = overall_performance_df['nnr'].fillna(1/overall_performance_df['precision'])
    overall_performance_df['nnr_adjusted'] = overall_performance_df['nnr_adjusted'].fillna(1/overall_performance_df['precision_adjusted'])

    overall_performance_df['database'] = overall_performance_df['database'].replace('oa', 'openalex')
    table1_col = ['Workflow', 'database']  + metric_col + ['nnr', 'n_retrieved'] + adjusted_metric_col
    overall_performance_df_final = overall_performance_df[table1_col].drop(columns = ['overall_screening_workload_adjusted'])
    return overall_performance_df_final, overarching_vs


@app.cell
def _(mo):
    mo.md("""
    ## Table 0 : Consolidated Performance and Efficiency Metrics of All Workflows
    """)
    return


@app.cell
def _(iqr, overall_performance_df_final):


    _col = ['recall', 'precision', 'f1_score', 'f2_score', 'f3_score', 'nnr', 'overall_screening_workload']
    _col_adjusted_metrics = [col + '_adjusted' for col in _col if col not in ['nnr', 'overall_screening_workload']]
    display_metric = _col + _col_adjusted_metrics

    # Calculate median and IQR separately
    workflow_grp_global_median = overall_performance_df_final[['Workflow'] + display_metric].groupby(['Workflow']).median()
    workflow_grp_global_iqr = overall_performance_df_final[['Workflow'] + display_metric].groupby(['Workflow']).agg(iqr)

    workflow_db_global_median = overall_performance_df_final[['Workflow', 'database'] + display_metric].groupby(['Workflow', 'database']).median()
    workflow_db_global_iqr = overall_performance_df_final[['Workflow', 'database'] + display_metric].groupby(['Workflow', 'database']).agg(iqr)
    return workflow_grp_global_iqr, workflow_grp_global_median


@app.cell
def _(overall_performance_df_final):
    _df = overall_performance_df_final[overall_performance_df_final['Workflow'].isin(['Workflow 1 (T)', 'Workflow 2 (O)'])]
    _df[['Workflow', 'database', 'recall', 'recall_adjusted']]
    return


@app.cell
def _(GT, pd, workflow_grp_global_iqr, workflow_grp_global_median):
    def tbl_0():
        # Define metric pairs
        metric_pairs = [
            ('recall', 'recall_adjusted'),
            ('precision', 'precision_adjusted'),
            ('f1_score', 'f1_score_adjusted'),
            ('f2_score', 'f2_score_adjusted'),
            ('f3_score', 'f3_score_adjusted')
        ]
        # other_metrics = ['nnr', 'overall_screening_workload']

        # First combine median and IQR into formatted strings
        workflow_grp_df = pd.DataFrame(index=workflow_grp_global_median.index)
        # workflow_db_grp_df = pd.DataFrame(index= workflow_db_global_median.index)

        def format_df(df_median, df_iqr, formatted_df):
            # Format paired metrics
            for raw, adjusted in metric_pairs:
                is_percentage = raw in ['recall', 'precision']

                if is_percentage:
                    formatted_df[raw] = [f"{(med*100):.1f}% ({(iqr*100):.1f}%)" 
                                       for med, iqr in zip(df_median[raw], df_iqr[raw])]
                    formatted_df[adjusted] = [f"{(med*100):.1f}% ({(iqr*100):.1f}%)" 
                                           for med, iqr in zip(df_median[adjusted], df_iqr[adjusted])]
                else:
                    formatted_df[raw] = [f"{med:.3f} ({iqr:.3f})" 
                                       for med, iqr in zip(df_median[raw], df_iqr[raw])]
                    formatted_df[adjusted] = [f"{med:.3f} ({iqr:.3f})" 
                                           for med, iqr in zip(df_median[adjusted], df_iqr[adjusted])]

            # Format other metrics
            # for metric in other_metrics:
            #     is_workload = 'workload' in metric
            #     if is_workload:
            #         formatted_df[metric] = [f"{med:,.0f} ({iqr:,.0f})" 
            #                              for med, iqr in zip(df_median[metric], df_iqr[metric])]
            #     else:
            #         formatted_df[metric] = [f"{med:.3f} ({iqr:.3f})" 
            #                              for med, iqr in zip(df_median[metric], df_iqr[metric])]

            return formatted_df

        workflow_grp_df = format_df(workflow_grp_global_median, workflow_grp_global_iqr, workflow_grp_df).reset_index()
        # workflow_db_grp_df = format_df(workflow_db_global_median, workflow_db_global_iqr, workflow_db_grp_df).reset_index()
        # separator_row_workflow_db = pd.DataFrame([{col: '—' for col in workflow_db_grp_df.columns}])
        # combined_df = pd.concat([workflow_grp_df, separator_row_workflow_db, workflow_db_grp_df], ignore_index=True)
        combined_df = workflow_grp_df


        # Create the table with new structure
        gt_tbl = (GT(combined_df)
                  .tab_header(title="Consolidated Workflow Performance Metrics - Whole-Of-Guideline Level")
                  # Add spanners for each metric pair
                  .tab_spanner(label="Recall", columns=['recall', 'recall_adjusted'])
                  .tab_spanner(label="Precision", columns=['precision', 'precision_adjusted'])
                  .tab_spanner(label="F1 Score", columns=['f1_score', 'f1_score_adjusted'])
                  .tab_spanner(label="F2 Score", columns=['f2_score', 'f2_score_adjusted'])
                  .tab_spanner(label="F3 Score", columns=['f3_score', 'f3_score_adjusted'])
                  # .tab_spanner(label="Efficiency Metrics", columns=other_metrics)
                  # Relabel columns
                  .cols_label(
                      Workflow="Workflow Type",
                      recall="Raw",
                      recall_adjusted="Adjusted*",
                      precision="Raw",
                      precision_adjusted="Adjusted*",
                      f1_score="Raw",
                      f1_score_adjusted="Adjusted*",
                      f2_score="Raw",
                      f2_score_adjusted="Adjusted*",
                      f3_score="Raw",
                      f3_score_adjusted="Adjusted*",
                      # nnr="NNR",
                      # overall_screening_workload="Workload"
                  ))

        #add footnotes 
        gt_tbl = gt_tbl.tab_source_note(
            source_note = "*Adjusted for underlying database coverage"
        )

        return gt_tbl


    table0 = tbl_0()
    table0
    return


@app.cell
def _(iqr, overall_performance_df_final):
    performance_metrics = ['recall', 'precision', 'n_retrieved']

    # Calculate database-specific medians and IQRs
    workflow_db_performance_median = overall_performance_df_final[['Workflow', 'database'] + performance_metrics].groupby(['database', 'Workflow']).median()
    workflow_db_performance_iqr = overall_performance_df_final[['Workflow', 'database'] + performance_metrics].groupby(['database', 'Workflow']).agg(iqr)
    workflow_db_performance_median
    return


@app.cell
def _(GT, iqr, overall_performance_df_final, pd):
    def tbl_1():
        # First get the metrics we want for this table
        performance_metrics = ['recall', 'precision', 'n_retrieved', 'nnr', 'overall_screening_workload']
        workflow_interest = ['Workflow 1 (T)', 'Workflow 2 (O)']

        # Calculate database-specific medians and IQRs
        workflow_db_performance_median = (
            overall_performance_df_final[
                overall_performance_df_final['Workflow'].isin(workflow_interest)
            ][['Workflow', 'database'] + performance_metrics]
            .groupby(['database', 'Workflow'])
            .median()
        )

        workflow_db_performance_iqr = (
            overall_performance_df_final[
                overall_performance_df_final['Workflow'].isin(workflow_interest)
            ][['Workflow', 'database'] + performance_metrics]
            .groupby(['database', 'Workflow'])
            .agg(iqr)
        )

        # Function to format values based on whether IQR is NaN or zero
        def format_values(med_recall, med_precision, med_retrieved, med_nnr, med_workload, iqr_val=None):
            if pd.isna(med_recall):
                return "N/A", "N/A", "N/A", "N/A"  # Return tuple for all metrics

            # Format recall with retrieved count
            recall_str = f"{med_recall*100:.1f}% (n={int(med_retrieved):,})"

            # Format precision
            if pd.isna(iqr_val) or iqr_val == 0:
                precision_str = f"{med_precision*100:.1f}%"
            else:
                precision_str = f"{med_precision*100:.1f}% ({iqr_val*100:.3f})†"

            # Format NNR and workload
            nnr_str = f"{med_nnr:.1f}"
            workload_str = f"{int(med_workload):,}"

            return recall_str, precision_str, nnr_str, workload_str

        # Format database-specific values
        formatted_db_df = pd.DataFrame(index=workflow_db_performance_median.index)

        for wf in workflow_interest:
            wf_data = workflow_db_performance_median.xs(wf, level='Workflow')
            wf_iqr = workflow_db_performance_iqr.xs(wf, level='Workflow')

            for idx in wf_data.index:
                recall_str, precision_str, nnr_str, workload_str = format_values(
                    wf_data.loc[idx, 'recall'],
                    wf_data.loc[idx, 'precision'],
                    wf_data.loc[idx, 'n_retrieved'],
                    wf_data.loc[idx, 'nnr'],
                    wf_data.loc[idx, 'overall_screening_workload'],
                    wf_iqr.loc[idx, 'precision'] if 'precision' in wf_iqr else None
                )
                formatted_db_df.loc[(idx, wf), 'recall'] = recall_str
                formatted_db_df.loc[(idx, wf), 'precision'] = precision_str
                formatted_db_df.loc[(idx, wf), 'nnr'] = nnr_str
                formatted_db_df.loc[(idx, wf), 'workload'] = workload_str

        # Reshape the data
        formatted_table = formatted_db_df.unstack()

        # Create final DataFrame
        final_df = pd.DataFrame(index=formatted_table.index)

        # Add columns and fill missing values with "N/A"
        for wf in workflow_interest:
            final_df[f"Recall_{wf}"] = formatted_table['recall', wf].fillna("N/A")
            final_df[f"Precision_{wf}"] = formatted_table['precision', wf].fillna("N/A")
            final_df[f"NNR_{wf}"] = formatted_table['nnr', wf].fillna("N/A")
            final_df[f"Workload_{wf}"] = formatted_table['workload', wf].fillna("N/A")

        final_df = final_df.reset_index()

        # Create column groups for metrics
        recall_columns = [f"Recall_{wf}" for wf in workflow_interest]
        precision_columns = [f"Precision_{wf}" for wf in workflow_interest]
        nnr_columns = [f"NNR_{wf}" for wf in workflow_interest]
        workload_columns = [f"Workload_{wf}" for wf in workflow_interest]

        gt_tbl = (GT(final_df)
                  .tab_header(title="Topic-Specific vs Overarching Search Performance: Whole-Of-Guideline Level")
                  .tab_spanner(
                      label="Recall (%, n)", 
                      columns=recall_columns
                  )
                  .tab_spanner(
                      label="Precision (%)", 
                      columns=precision_columns
                  )
                  .tab_spanner(
                      label="Number Needed to Read", 
                      columns=nnr_columns
                  )
                  .tab_spanner(
                      label="Screening Workload", 
                      columns=workload_columns
                  )
                  .cols_label(
                      database="Database",
                      **{f"Recall_{wf}": wf for wf in workflow_interest},
                      **{f"Precision_{wf}": wf for wf in workflow_interest},
                      **{f"NNR_{wf}": wf for wf in workflow_interest},
                      **{f"Workload_{wf}": wf for wf in workflow_interest}
                  )
                  .tab_source_note(
                      source_note="T:Single Database Topic Specific Search, O: Single Database Overarching Search"
                  ))

        return gt_tbl

    return (tbl_1,)


@app.cell
def _():
    # Result Data for Table 1
    return


@app.cell
def _(tbl_1):
    table1 = tbl_1()
    table1
    return


@app.cell
def _():
    # Result Figure 1
    return


@app.cell
def _(overarching_vs, pd, plt, sns):
    def create_clean_nnr_plot():
        # First, set up your data as before
        vs_overarching_extract = overarching_vs[['vector_search_type','nnr_cutoff', 'nnr_raw', 'database']].copy()
        vs_overarching_extract['nnr_change'] = vs_overarching_extract['nnr_cutoff'] - (vs_overarching_extract['nnr_raw'])  
        vs_overarching_extract['nnr_change_pct'] = vs_overarching_extract['nnr_change'] / vs_overarching_extract['nnr_raw'] * 100

        # Set the vector search type order
        vector_type_order = ["zero-shot","one-shot",  "few-shot"]
        vs_overarching_extract['vector_search_type'] = pd.Categorical(
            vs_overarching_extract['vector_search_type'], 
            categories=vector_type_order, 
            ordered=True
        )

        # Create a more minimal, ggplot-like style
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create the plot
        vstype_overarching_bar_plot = sns.barplot(
            x='database', 
            y='nnr_change', 
            hue='vector_search_type',
            hue_order=vector_type_order,
            data=vs_overarching_extract,
            palette='Set2',
            alpha=0.9,
            dodge=True,
            ax=ax, 
            errorbar=None
        )

        # Get baseline values for each database
        baseline_values = vs_overarching_extract.groupby('database')['nnr_raw'].first().to_dict()

        # OPTION 1: Add baseline info to x-axis labels
        current_labels = [label.get_text() for label in ax.get_xticklabels()]
        new_labels = [f'{label}\n(Baseline (W2): {baseline_values[label]:.1f})' 
                      for label in current_labels]
        ax.set_xticklabels(new_labels, fontsize=10)

        # Add percentage annotations on top of each bar
        for container in vstype_overarching_bar_plot.containers:
            vstype_overarching_bar_plot.bar_label(container, fmt='%.1f', padding=3, fontsize=8)

        for spine in ax.spines.values():
            spine.set_visible(False)
        # Customize the grid
        ax.grid(axis='both', color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)  # Place gridlines behind bars

        # Set background
        ax.set_facecolor('white')

        # Add title and labels
        plt.title('Absolute Reduction in NNR between W3 from Baseline (W2) \n by Database and Prior Knowledge Scenario', fontsize=14)
        plt.xlabel('Database', fontsize=12)
        plt.ylabel('NNR Reduction', fontsize=12)

        # Set legend
        plt.legend(title='Prior Knowledge Scenario', frameon=True)

        # Adjust layout
        plt.subplots_adjust(top=0.9)
        plt.tight_layout()

        # Show the plot
        plt.show()

        return vs_overarching_extract

    # Use this version - it's the cleanest
    vs_overarching_nnr_comparison = create_clean_nnr_plot()
    return (vs_overarching_nnr_comparison,)


@app.cell
def _(vs_overarching_nnr_comparison):
    vs_overarching_nnr_comparison
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Topic Level Comparisons
    """)
    return


@app.cell
def _(pd, topicspecific_novs_results_df, topicspecific_vs_results_df):
    #grab one vector search scenario to prevent double / triple counting of recall as they all come from the same umderlying search, just refined 
    o_tvs_zeroshot = topicspecific_vs_results_df[topicspecific_vs_results_df['vector_search_type'] == 'few-shot'].copy()
    o_tvs_zeroshot['Workflow'] = 'Workflow 2 (O)'
    #combine with no vector search results 
    recall_comparison_t_o = pd.concat([topicspecific_novs_results_df, o_tvs_zeroshot])
    recall_comparison_t_o['recall_pct'] = recall_comparison_t_o['recall'] * 100
    recall_comparison_t_o = recall_comparison_t_o[recall_comparison_t_o['question_id'] != 'overall']
    return o_tvs_zeroshot, recall_comparison_t_o


@app.cell
def _(recall_comparison_t_o):
    count_col = 'n_retrieved'
    _grouping_cols = ['database', 'Workflow']
    zero_retrieved = recall_comparison_t_o[recall_comparison_t_o[count_col] == 0]
    zero_counts = zero_retrieved.groupby(_grouping_cols).size()
    zero_counts_table = zero_counts.unstack(level='Workflow', fill_value=0)
    zero_counts_table
    return


@app.cell
def _(mo):
    mo.md("""
    ## Topic Level Recall, Topic Specific vs Overarching Workflows
    """)
    return


@app.cell
def _(np, pd, recall_comparison_t_o, stats):
    recall_comparison_t_o['standardised_question_id'] = recall_comparison_t_o['question_id'].str.replace('/', ',')
    recall_pval_df = pd.DataFrame()
    paired_t_o_comparison_df = pd.DataFrame()

    for d in ['embase', 'medline']: 
        w1 = recall_comparison_t_o.query('Workflow == "Workflow 1 (T)" & database == @d')
        w2 = recall_comparison_t_o.query('Workflow == "Workflow 2 (O)"& database == @d')
        topic_column = 'question_id'
        w1 = w1.copy()
        w2 = w2.copy()
        w1['std_id'] = w1[topic_column].str.replace('/', ',')
        w2['std_id'] = w2[topic_column].str.replace('/', ',')
        w1 = w1[w1['std_id'] != 'overall']
        w2 = w2[w2['std_id'] != 'overall']
        common_topics = set(w1['std_id']) & set(w2['std_id'])

        w1_filtered = w1[w1['std_id'].isin(common_topics)].sort_values(by='std_id')
        w2_filtered = w2[w2['std_id'].isin(common_topics)].sort_values(by='std_id')
        # Wilcoxon tests (paired)
        raw_recall_statistic, raw_recall_pval = stats.wilcoxon(
            w1_filtered['recall_pct'], w2_filtered['recall_pct'], alternative='two-sided')
        adj_recall_statistic, adj_recall_pval = stats.wilcoxon(
            w1_filtered['recall_adjusted'], w2_filtered['recall_adjusted'], alternative='two-sided')

        # Paired differences
        paired_diff_raw = w2_filtered['recall_pct'].values - w1_filtered['recall_pct'].values
        paired_diff_adj = w2_filtered['recall_adjusted'].values*100 - w1_filtered['recall_adjusted'].values*100

        # Median and IQR
        w1_median_raw = np.median(w1_filtered['recall_pct'])
        w1_iqr_raw = np.percentile(w1_filtered['recall_pct'], 75) - np.percentile(w1_filtered['recall_pct'], 25)
        w2_median_raw = np.median(w2_filtered['recall_pct'])
        w2_iqr_raw = np.percentile(w2_filtered['recall_pct'], 75) - np.percentile(w2_filtered['recall_pct'], 25)

        w1_median_adj = np.median(w1_filtered['recall_adjusted'])*100
        w1_iqr_adj = np.percentile(w1_filtered['recall_adjusted'], 75)*100 - np.percentile(w1_filtered['recall_adjusted'], 25)*100
        w2_median_adj = np.median(w2_filtered['recall_adjusted'])*100
        w2_iqr_adj = np.percentile(w2_filtered['recall_adjusted'], 75)*100 - np.percentile(w2_filtered['recall_adjusted'], 25)*100    
    
        median_diff_raw = np.median(paired_diff_raw)
        iqr_diff_raw = np.percentile(paired_diff_raw, 75) - np.percentile(paired_diff_raw, 25)
        median_diff_adj = np.median(paired_diff_adj)
        iqr_diff_adj = np.percentile(paired_diff_adj, 75) - np.percentile(paired_diff_adj, 25)

        # Format p-values
        raw_formatted = '<0.01' if raw_recall_pval <= 0.01 else f"{raw_recall_pval:.6f}"
        adj_formatted = '<0.01' if adj_recall_pval <= 0.01 else f"{adj_recall_pval:.6f}"

        print(f'Database : {d}')
        print('---')
        print(f'W1 Recall (raw): {w1_median_raw} IQR: {w1_iqr_raw}')
        print(f'W2 Recall (raw): {w2_median_raw} IQR: {w2_iqr_raw}')
        print(f'Median paired diff (raw): {median_diff_raw}, IQR: {iqr_diff_raw}')
        print(f'Raw recall stat: {raw_recall_statistic}, raw recall p: {raw_formatted}')
        print(f'--')
        print(f'W1 Recall (adj): {w1_median_adj} IQR: {w1_iqr_adj}')
        print(f'W2 Recall (adj): {w2_median_adj} IQR: {w2_iqr_adj}')
        print(f'Adj recall stat: {adj_recall_statistic}, adj recall p: {adj_formatted}')
        print(f'Median paired diff (adj): {median_diff_adj}, IQR: {iqr_diff_adj}')
        print('---')

        result_dct = {
            'database': d,
            'raw_recall_pval': raw_formatted,
            'adj_recall_pval': adj_formatted,
            'median_diff_raw': median_diff_raw,
            'iqr_diff_raw': iqr_diff_raw,
            'median_diff_adj': median_diff_adj,
            'iqr_diff_adj': iqr_diff_adj, 
        }
        recall_pval_df = pd.concat([recall_pval_df, pd.DataFrame([result_dct])], ignore_index=True)

    recall_pval_df
    return


@app.cell
def _(pd, topicspecific_novs_results_df, topicspecific_vs_results_df):
    topicspecific_vs_results_df['Workflow'] = 'Workflow 4 (O+TVS)'
    topicspecific_vs_extract = topicspecific_vs_results_df[['database', 'question_id', 'gdg', 'sr_type','Workflow','n_retrieved', 'n_missed','recall','recall_adjusted','nnr_cutoff', 'precision_cutoff', 'overall_screening_workload_cutoff', 'f1_score_cutoff', 'f3_score_cutoff','vector_search_type']].copy()
    topicspecific_vs_extract['standardised_question_id'] = topicspecific_vs_extract['question_id'].str.replace('/', ',')
    #rename 
    topicspecific_vs_extract.rename(columns = {
        'nnr_cutoff' : 'nnr', 
        'precision_cutoff' : 'precision', 
        'overall_screening_workload_cutoff' : 'overall_screening_workload', 
        'f1_score_cutoff' : 'f1_score', 
        'f3_score_cutoff' : 'f3_score'
    } ,inplace=True)



    topicspecific_prevs_integration = topicspecific_vs_results_df[['n_retrieved', 'n_missed','recall', 'recall_adjusted', 'f1_score', 'f3_score', 'overall_screening_workload', 'question_id', 'gdg', 'sr_type', 'database', 'vector_search_type']].copy().query("vector_search_type == 'zero-shot'")
    topicspecific_prevs_integration['Workflow'] = 'Workflow 2 (O)'
    topicspecific_prevs_integration['vector_search_type'] = 'none'
    topicspecific_prevs_integration['nnr'] = topicspecific_vs_results_df['nnr_raw']
    topicspecific_prevs_integration['precision_pct'] =  topicspecific_vs_results_df['precision']


    topicspecific_novs_results_df['standardised_question_id'] =  topicspecific_novs_results_df['question_id'].str.replace('/', ',')
    topicspecific_novs_extract = topicspecific_novs_results_df[['standardised_question_id', 'question_id', 'gdg', 'sr_type','n_retrieved', 'n_missed','recall','recall_adjusted', 'database', 'Workflow', 'nnr', 'precision', 'overall_screening_workload', 'f1_score', 'f3_score']].copy()

    topic_comparisons = pd.concat([topicspecific_novs_extract, topicspecific_vs_extract])
    topic_comparisons['precision_pct'] = topic_comparisons['precision']*100


    # topic_comparisons = pd.concat([topic_comparisons,topicspecific_prevs_integration])

    topic_comparisons['recall_pct'] = topic_comparisons['recall']*100
    topic_comparisons['recall_adjusted_pct'] = topic_comparisons['recall_adjusted']*100
    topic_comparisons['n_relevant'] = topic_comparisons['n_retrieved'] + topic_comparisons['n_missed']
    topic_comparison = topic_comparisons[topic_comparisons['question_id']!= 'overall']

    # Ensure the column is categorical with the specified order
    topic_comparison['vector_search_type'] = pd.Categorical(
        topic_comparison['vector_search_type'],
        categories=['zero-shot', 'one-shot', 'few-shot', 'none'],
        ordered=False
    )
    topic_comparison['vector_search_type'] = topic_comparison['vector_search_type'].fillna('none')
    topic_comparison['workflow_group'] = (
        topic_comparison['Workflow'].astype(str) + ' + ' + topic_comparison['vector_search_type'].astype(str)
    )
    return topic_comparison, topic_comparisons, topicspecific_prevs_integration


@app.cell
def _():
    ## Topic Level Summary Stats
    return


@app.cell
def _(topic_comparison):
    topic_comparison
    return


@app.cell
def _(iqr, topic_comparison):
    _metric = 'precision'
    grouping_cols = ['database', 'workflow_group']


    print(f"Calculating Median and IQR for '{_metric}' grouped by {grouping_cols}...")

    # Group by database and workflow, then apply aggregations to the specific metric column
    topic_comparison_summary = topic_comparison.groupby(grouping_cols).agg(
        # Create new columns: 'Median' calculated from _metric using 'median'
        Median=(_metric, 'median'),
        # Create new column: 'IQR' calculated from _metric using the custom iqr function
        IQR=(_metric, iqr)
    )

    topic_comparison_summary
    return


@app.cell
def _(topic_comparison):
    topic_comparison
    return


@app.cell
def _(topic_comparison):
    topic_comparison.to_parquet("topic_comparison.parquet")
    return


@app.cell
def _():
    ## Topic Specific Vector Search 
    return


@app.cell
def _(pd, topic_comparison, topicspecific_prevs_integration):
    w4_data = topic_comparison.query('Workflow == "Workflow 4 (O+TVS)"')
    topicspecific_prevs_integration['nnr_overarching'] = topicspecific_prevs_integration['nnr']
    topicspecific_prevs_integration['standardised_question_id'] =  topicspecific_prevs_integration['question_id'].str.replace('/', ',')

    w4_data_nnrchange = w4_data_nnrchange = pd.merge(
        w4_data,
        topicspecific_prevs_integration[['standardised_question_id', 'nnr_overarching', 'database']], 
        on=['standardised_question_id', 'database']
    ).query('nnr_overarching != 0')


    w4_data_nnrchange['nnr_change'] = w4_data_nnrchange['nnr_overarching'] - w4_data_nnrchange['nnr']
    w4_data_nnrchange['nnr_change_pct'] = (w4_data_nnrchange['nnr_change'] / w4_data_nnrchange['nnr_overarching'])*100
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Influence of Vector Search / Ranking
    """)
    return


@app.cell
def _(o_tvs_zeroshot, pd, topic_comparisons):
    w1_topicspecifc_precutoff = o_tvs_zeroshot[['precision', 'nnr_raw', 'f3_score','database', 'question_id']].copy()
    w1_topicspecifc_precutoff['Workflow'] = 'Workflow 2 (O)'
    w1_topicspecifc_precutoff.rename(columns = {'nnr_raw': 'nnr'}, inplace=True)

    vs_scenario = 'few-shot'
    vs_scenario_df = topic_comparisons.query("vector_search_type == @vs_scenario")

    w1_specific_data = topic_comparisons.query('Workflow == "Workflow 1 (T)"')


    topic_level_nnr_comparison_df = pd.concat([w1_specific_data,w1_topicspecifc_precutoff, vs_scenario_df[['Workflow', 'database','precision', 'nnr', 'f3_score', 'question_id']]])
    topic_level_nnr_comparison_df = topic_level_nnr_comparison_df[topic_level_nnr_comparison_df['question_id'] != 'overall']
    topic_level_nnr_comparison_df['standardised_question_id'] = topic_level_nnr_comparison_df['question_id'].str.replace('/', ',')
    topic_level_nnr_comparison_df.drop(columns = ['question_id'], inplace=True)
    topic_level_nnr_comparison_df['database'] = topic_level_nnr_comparison_df['database'].str.capitalize()
    return (topic_level_nnr_comparison_df,)


@app.cell
def _(topic_level_nnr_comparison_df):
    topic_level_nnr_comparison_df
    return


@app.cell
def _():
    # Vector Search Comparisons (Whole of Guideline)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Prep Vector Search Comparisons (Topic Level)
    """)
    return


@app.cell
def _(topicspecific_vs_results_df):
    topicspecific_vs_results_df
    return


@app.cell
def _(topicspecific_vs_results_df):
    w4_vs_excerpt = topicspecific_vs_results_df.copy()
    w4_vs_excerpt['nnr'] = w4_vs_excerpt['nnr_cutoff']
    w4_vs_excerpt['overall_screening_workload'] = w4_vs_excerpt['overall_screening_workload_cutoff']
    return (w4_vs_excerpt,)


@app.cell
def _(pd, topicspecific_prevs_integration, w4_vs_excerpt):
    col_extract_vs_comparison = ['standardised_question_id', 'nnr', 'overall_screening_workload','vector_search_type','Workflow', 'database']

    vs_comparison = pd.concat([topicspecific_prevs_integration[col_extract_vs_comparison], w4_vs_excerpt[col_extract_vs_comparison]])
    return (vs_comparison,)


@app.cell
def _(vs_comparison):
    vs_comparison.to_csv('vs_comparison.csv')
    return


if __name__ == "__main__":
    app.run()
