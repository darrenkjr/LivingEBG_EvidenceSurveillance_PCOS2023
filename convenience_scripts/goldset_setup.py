import pandas as pd 
from pathlib import Path


def goldset_setup():
    current_dir = Path(__file__).parent
    dataset_dir = current_dir.parent / 'dataset'
    groundtruth_path = dataset_dir / 'fullgroundtruth_valid_apimerge_df.parquet'
    #setup gold set articles to inform vector search and generating topics for openalex search 
    fullgroundtruth_valid_apimerge_df = pd.read_parquet(groundtruth_path)
    fullgroundtruth_valid_apimerge_df['year_pub_extract'] = pd.to_numeric(
        fullgroundtruth_valid_apimerge_df['year_pub_extract'], 
        errors='coerce'
    ).astype('Int64')  

    fullgroundtruth_valid_apimerge_df.reset_index(inplace = True)

    #firstly for sr updates
    _srupdate = fullgroundtruth_valid_apimerge_df.query('sr_update == "Y" & title.notna() & abstract.notna()').copy()
    _newsr = fullgroundtruth_valid_apimerge_df.query('sr_update != "Y" & title.notna() & abstract.notna()').copy()
    srupdate_goldset = _srupdate.query('year_pub_extract <= searchstrat_year_start').copy()

    #now, new SRs - choose random as long sa title and abstracts are not empty
    newsr_goldset = _newsr.groupby('question_id').apply(
        lambda x: x.sample(
            n=min(3, len(x)),  # x is already filtered for non-empty title/abstract
            replace=False,
            random_state=42
        )
    ).reset_index(drop=True)

    combined_goldset = pd.concat([srupdate_goldset, newsr_goldset], ignore_index = True)
    #export 
    combined_goldset.to_parquet(dataset_dir / 'combined_goldset.parquet')
    combined_goldset.to_excel(dataset_dir / 'combined_goldset.xlsx', index=False)
    return combined_goldset


goldset_setup()