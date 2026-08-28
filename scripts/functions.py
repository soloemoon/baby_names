import re
from pathlib import Path
import polars as pl
import yaml

def clean_file_path(
    dir: str,
    folder: str,
    filename: str
) -> str:
    """
    Cleans the given file path by removing any leading/trailing whitespace and normalizing the path.
    
    Parameters:
    - dir (str): The base directory path.
    - folder (str): The folder name within the base directory.
    - filename (str): The name of the file.

    Returns:
    - str: The cleaned and normalized file path.
    """
      
    return str(Path(dir) / folder / filename)

def combine_all_txt_files_df(
    directory_path: str,
    dataset_type: str = "year"
) -> pl.DataFrame:
    """
    Combines all .txt files in the specified directory into a single Polars DataFrame.

    Assumes files follow the SSA baby names format (e.g. yob1990.txt) with
    unheaded columns: name, sex, count. A `year` column is extracted from
    each filename (first run of 4 digits) and added to the output.

    Args:
        directory_path (str): The path to the directory containing .txt files.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the combined data from all .txt files.

    Raises:
        FileNotFoundError: If no .txt files are found in the directory.
    """
    txt_paths = sorted(Path(directory_path).glob("*.txt"))

    if not txt_paths:
        raise FileNotFoundError(f"No .txt files found in {directory_path}")
    
    if dataset_type not in ["year", "state"]:
        raise ValueError("dataset_type must be either 'year' or 'state'")

    if dataset_type == "year":
        new_columns = ["name", "sex", "count"]
        override ={"name": pl.Utf8, "sex": pl.Utf8, "count": pl.Int64}
    else:  # dataset_type == "state"
        new_columns = ["state", "sex", "year", "name", "count"]
        override ={"state": pl.Utf8, "sex": pl.Utf8, "year": pl.Int64, "name": pl.Utf8, "count": pl.Int64}

    frames = []
    for path in txt_paths:
        df = pl.read_csv(
            path,
            has_header=False,
            new_columns=new_columns,
            schema_overrides=override,
        )
        
        # For year-based files, extract year from filename and add as column
        # For state files, year is already in the CSV data (column 3)
        if dataset_type == 'year':
            year_match = re.search(r"(\d{4})", path.stem)
            year = int(year_match.group(1)) if year_match else None
            df = df.with_columns(pl.lit(year).alias("year"))
        
        frames.append(df)

    return pl.concat(frames, how="vertical")


def get_top_names_by_year(
    df: pl.DataFrame, rank: int = 1, top_n: int = 1
) -> pl.DataFrame:
    """
    Get the nth most popular name(s) for each sex by year.

    Args:
        df (pl.DataFrame): DataFrame with columns: name, sex, count, year
        rank (int): Which rank to retrieve (1 = most popular, 2 = second most, etc.).
                   Defaults to 1.
        top_n (int): How many top names to retrieve (e.g., top_n=5 gets ranks 1-5).
                    Defaults to 1. If provided, overrides rank parameter.

    Returns:
        pl.DataFrame: DataFrame with the requested rank(s) for each year/sex combination,
                     including a 'rank' column.
    """
    if top_n > 1:
        # Get multiple ranks
        result = (
            df.sort("year", "sex", "count", descending=[False, False, True])
            .with_columns(
                pl.col("count")
                .rank(method="ordinal", descending=True)
                .over("year", "sex")
                .alias("rank")
            )
            .filter(pl.col("rank") <= top_n)
            .sort("year", "sex", "rank")
        )
    else:
        # Get single rank
        result = (
            df.sort("year", "sex", "count", descending=[False, False, True])
            .with_columns(
                pl.col("count")
                .rank(method="ordinal", descending=True)
                .over("year", "sex")
                .alias("rank")
            )
            .filter(pl.col("rank") == rank)
            .sort("year", "sex")
        )

    return result

    
def generate_file_paths_yaml(
   file_path_dict: dict, 
    yaml_output_path: str
) -> dict:
    """
    Generates the project's key file paths and writes them to a YAML file.

    Args:
        file_path_dict (dict): A dictionary containing the file paths.
        yaml_output_path (str): The path where the YAML file will be written.

    Returns:
        dict: The dictionary of file paths that was written to the YAML file.
    """
   
    with open(yaml_output_path, "w") as f:
        yaml.safe_dump(file_path_dict, f, sort_keys=False)

        return file_path_dict

def export_polars_df_to_parquet(
    df: pl.DataFrame,
     output_path: str
) -> None:
    """
    Exports a Polars DataFrame to a Parquet file.

    Args:
        df (pl.DataFrame): The Polars DataFrame to export.
        output_path (str): The path where the Parquet file will be saved.
    """
    df.write_parquet(output_path)

# Probability of having a given name
def compute_name_probability(
    df: pl.DataFrame,
    name: str, 
    sex: str = None, 
    year: int = None
) -> dict:
    """
    Compute the probability someone will have a given name.
    
    Args:
        name (str): The name to look up
        sex (str): Optional - 'M' or 'F' to filter by sex
        year (int): Optional - specific year to analyze
    
    Returns:
        dict: Dictionary with probability and supporting statistics
    """
    # Filter data
    filtered = df.filter(pl.col("name") == name)
    
    if sex:
        filtered = filtered.filter(pl.col("sex") == sex)
    
    if year:
        filtered = filtered.filter(pl.col("year") == year)
    
    # Calculate statistics
    name_count = filtered.get_column("count").sum()
    
    # Get total for comparison
    comparison_data = df
    if sex:
        comparison_data = comparison_data.filter(pl.col("sex") == sex)
    if year:
        comparison_data = comparison_data.filter(pl.col("year") == year)
    
    total_count = comparison_data.get_column("count").sum()
    
    # Calculate probability
    probability = (name_count / total_count) if total_count > 0 else 0
    
    return {
        "name": name,
        "sex": sex if sex else "All",
        "year": year if year else "All Years",
        "count": name_count,
        "total": total_count,
        "probability": probability,
        "percentage": probability * 100,
        "odds": f"1 in {int(1/probability):,}" if probability > 0 else "N/A"
    }
