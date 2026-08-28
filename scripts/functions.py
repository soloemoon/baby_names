import re
from pathlib import Path
import polars as pl
import yaml
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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

def compute_name_probability(df, name: str, sex: str = None, year: int = None) -> dict:
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


def plot_names_and_births_by_year(
    df: pl.DataFrame,
    height: int = 800
) -> go.Figure:
    """
    Create a subplot with two charts:
    1. Number of distinct names by year and sex (grouped bars)
    2. Total births by year and sex (stacked bars)
    
    Args:
        df (pl.DataFrame): DataFrame with columns: name, sex, count, year
        height (int): Height of the figure in pixels. Defaults to 800.
    
    Returns:
        go.Figure: Plotly figure object with the two subplots
    """
    # Calculate distinct names per year
    names_per_year = (
        df
        .group_by("year", "sex")
        .agg(pl.col("name").n_unique().alias("n_names"))
        .sort("year")
    )
    
    # Calculate total births per year
    births_by_year = (
        df
        .group_by("year", "sex")
        .agg(pl.col("count").sum().alias("total_births"))
        .sort("year")
    )
    
    # Create subplot with both charts
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "Number of Distinct Names by Year and Sex",
            "Total Births per Year by Sex"
        ),
        vertical_spacing=0.12
    )
    
    # Add first chart (distinct names - grouped bars)
    for sex in ["M", "F"]:
        data = names_per_year.filter(pl.col("sex") == sex)
        fig.add_trace(
            go.Bar(
                x=data.get_column("year"),
                y=data.get_column("n_names"),
                name=sex,
                legendgroup=sex,
                marker_color="#1f77b4" if sex == "M" else "#ff7f0e",
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Add second chart (total births - stacked bars)
    for sex in ["M", "F"]:
        data = births_by_year.filter(pl.col("sex") == sex)
        fig.add_trace(
            go.Bar(
                x=data.get_column("year"),
                y=data.get_column("total_births"),
                name=sex,
                legendgroup=sex,
                marker_color="#1f77b4" if sex == "M" else "#ff7f0e",
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=height,
        barmode="stack",
        hovermode="x unified",
        legend=dict(
            title="Sex",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update x-axes labels
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=1)
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Number of Distinct Names", row=1, col=1)
    fig.update_yaxes(title_text="Total Births", row=2, col=1)
    
    # Make first subplot grouped instead of stacked
    fig.update_traces(row=1, col=1, selector=dict(type='bar'))
    fig.data[0].update(offsetgroup=0)
    fig.data[1].update(offsetgroup=1)
    
    return fig


def plot_diversity_rate_trends(
    df: pl.DataFrame,
    time_periods: list,
    height: int = 500
) -> go.Figure:
    """
    Create a subplot showing diversity rate trends over time:
    1. Names per 1,000 births over time (line chart)
    2. Change in diversity rate between periods (bar chart)
    
    Args:
        df (pl.DataFrame): DataFrame with columns: name, sex, count, year
        time_periods (list): List of tuples (start_year, end_year, period_label)
                            e.g., [(1880, 1919, "1880-1919"), (1920, 1959, "1920-1959")]
        height (int): Height of the figure in pixels. Defaults to 500.
    
    Returns:
        go.Figure: Plotly figure object with the two subplots
    """
    # Calculate diversity rate for each period
    diversity_rate_results = []
    
    for start_year, end_year, period_label in time_periods:
        period_data = df.filter(
            (pl.col("year") >= start_year) & (pl.col("year") <= end_year)
        )
        
        for sex_val in ["M", "F"]:
            sex_data = period_data.filter(pl.col("sex") == sex_val)
            
            # Total births and distinct names for the period
            total_births = sex_data.get_column("count").sum()
            total_names = sex_data.get_column("name").n_unique()
            
            # Average per year to get annual rates
            n_years = end_year - start_year + 1
            avg_births_per_year = total_births / n_years
            avg_names_per_year = total_names / n_years
            
            # Names per 1,000 births
            names_per_1k_births = (total_names / total_births) * 1000
            
            diversity_rate_results.append({
                "period": period_label,
                "sex": sex_val,
                "total_births": total_births,
                "total_distinct_names": total_names,
                "avg_births_per_year": round(avg_births_per_year, 0),
                "avg_names_per_year": round(avg_names_per_year, 0),
                "names_per_1k_births": round(names_per_1k_births, 2)
            })
    
    diversity_rate_df = pl.DataFrame(diversity_rate_results)
    
    # Create subplot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Names per 1,000 Births Over Time",
            "Change in Diversity Rate Between Periods"
        ),
        horizontal_spacing=0.15
    )
    
    # Line chart of diversity rate
    for sex_val in ["M", "F"]:
        data = diversity_rate_df.filter(pl.col("sex") == sex_val)
        fig.add_trace(
            go.Scatter(
                x=data.get_column("period"),
                y=data.get_column("names_per_1k_births"),
                name=sex_val,
                legendgroup=sex_val,
                marker_color="#1f77b4" if sex_val == "M" else "#ff7f0e",
                mode="lines+markers",
                line=dict(width=3)
            ),
            row=1, col=1
        )
    
    # Calculate period-over-period change
    for sex_val in ["M", "F"]:
        sex_df = diversity_rate_df.filter(pl.col("sex") == sex_val).sort("period")
        rates = sex_df.get_column("names_per_1k_births").to_list()
        periods = sex_df.get_column("period").to_list()
        
        # Calculate changes
        changes = [rates[i] - rates[i-1] if i > 0 else 0 for i in range(len(rates))]
        
        fig.add_trace(
            go.Bar(
                x=periods,
                y=changes,
                name=sex_val,
                legendgroup=sex_val,
                marker_color="#1f77b4" if sex_val == "M" else "#ff7f0e",
                showlegend=False
            ),
            row=1, col=2
        )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=2)
    
    fig.update_layout(
        height=height,
        hovermode="x unified",
        barmode="group"
    )
    
    fig.update_xaxes(title_text="Time Period", row=1, col=1)
    fig.update_xaxes(title_text="Time Period", row=1, col=2)
    fig.update_yaxes(title_text="Names per 1,000 Births", row=1, col=1)
    fig.update_yaxes(title_text="Change from Previous Period", row=1, col=2)
    
    return fig


def plot_names_births_correlation(
    correlation_data: pl.DataFrame,
    correlations: pl.DataFrame,
    sex_colors: dict = None,
    height: int = 500
) -> go.Figure:
    """
    Create scatter plots showing the correlation between number of distinct names
    and total births by sex.
    
    Args:
        correlation_data (pl.DataFrame): DataFrame with columns: year, sex, total_births, n_names
        correlations (pl.DataFrame): DataFrame with columns: sex, correlation
        sex_colors (dict): Optional dict mapping sex to colors. Defaults to {"M": "#1f77b4", "F": "#ff7f0e"}
        height (int): Height of the figure in pixels. Defaults to 500.
    
    Returns:
        go.Figure: Plotly figure object with correlation scatter plots
    """
    if sex_colors is None:
        sex_colors = {"M": "#1f77b4", "F": "#ff7f0e"}
    
    # Create subplots with correlation values in titles
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Males (r = {correlations.filter(pl.col('sex') == 'M').get_column('correlation')[0]:.3f})",
            f"Females (r = {correlations.filter(pl.col('sex') == 'F').get_column('correlation')[0]:.3f})"
        )
    )
    
    # Male scatter plot
    male_data = correlation_data.filter(pl.col("sex") == "M")
    fig.add_trace(
        go.Scatter(
            x=male_data.get_column("total_births"),
            y=male_data.get_column("n_names"),
            mode="markers",
            marker=dict(color=sex_colors["M"], size=8, opacity=0.6),
            name="Male",
            hovertemplate="Births: %{x:,}<br>Distinct Names: %{y:,}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Female scatter plot
    female_data = correlation_data.filter(pl.col("sex") == "F")
    fig.add_trace(
        go.Scatter(
            x=female_data.get_column("total_births"),
            y=female_data.get_column("n_names"),
            mode="markers",
            marker=dict(color=sex_colors["F"], size=8, opacity=0.6),
            name="Female",
            hovertemplate="Births: %{x:,}<br>Distinct Names: %{y:,}<extra></extra>"
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=height,
        showlegend=False,
        title_text="Correlation: Number of Distinct Names vs Total Births by Sex"
    )
    
    fig.update_xaxes(title_text="Total Births", row=1, col=1)
    fig.update_xaxes(title_text="Total Births", row=1, col=2)
    fig.update_yaxes(title_text="Number of Distinct Names", row=1, col=1)
    fig.update_yaxes(title_text="Number of Distinct Names", row=1, col=2)
    
    return fig


def plot_interactive_top_names_comparison(
    df: pl.DataFrame,
    decade_start: int = 1880,
    decade_end: int = 2020,
    decade_step: int = 10,
    default_left_decade: int = 1950,
    default_right_decade: int = 2020,
    default_left_sex: str = "M",
    default_right_sex: str = "M",
    sex_colors: dict = None,
    height: int = 600,
    top_n: int = 10
) -> go.Figure:
    """
    Create an interactive side-by-side comparison of top names by decade and sex
    with dropdown filters.
    
    Args:
        df (pl.DataFrame): DataFrame with columns: name, sex, count, year
        decade_start (int): First decade to include (e.g., 1880). Defaults to 1880.
        decade_end (int): Last decade to include (e.g., 2020). Defaults to 2020.
        decade_step (int): Step between decades (e.g., 10 for each decade). Defaults to 10.
        default_left_decade (int): Default decade for left table. Defaults to 1950.
        default_right_decade (int): Default decade for right table. Defaults to 2020.
        default_left_sex (str): Default sex for left table ('M' or 'F'). Defaults to 'M'.
        default_right_sex (str): Default sex for right table ('M' or 'F'). Defaults to 'M'.
        sex_colors (dict): Optional dict mapping sex to colors. Defaults to {"M": "#1f77b4", "F": "#ff7f0e"}
        height (int): Height of the figure in pixels. Defaults to 600.
        top_n (int): Number of top names to display. Defaults to 10.
    
    Returns:
        go.Figure: Plotly figure object with interactive tables
    """
    if sex_colors is None:
        sex_colors = {"M": "#1f77b4", "F": "#ff7f0e"}
    
    # Prepare data for all decades and both sexes
    all_decades = list(range(decade_start, decade_end + decade_step, decade_step))
    interactive_data = {}
    
    for decade_start_val in all_decades:
        decade_end_val = decade_start_val + decade_step - 1
        
        decade_data = df.filter(
            (pl.col("year") >= decade_start_val) & (pl.col("year") <= decade_end_val)
        )
        
        if decade_data.height == 0:
            continue
        
        for sex_val in ["M", "F"]:
            sex_data = decade_data.filter(pl.col("sex") == sex_val)
            
            if sex_data.height == 0:
                continue
            
            # Get top N names
            top_names = (
                sex_data
                .group_by("name")
                .agg(pl.col("count").sum().alias("total_count"))
                .sort("total_count", descending=True)
                .head(top_n)
            )
            
            total_births = sex_data.get_column("count").sum()
            top_names = top_names.with_columns(
                (pl.col("total_count") / total_births * 100).alias("pct_of_births")
            )
            
            key = f"{decade_start_val}s_{sex_val}"
            interactive_data[key] = top_names
    
    # Helper function to create table data
    def create_table_trace(data_df, color):
        return go.Table(
            header=dict(
                values=["<b>Rank</b>", "<b>Name</b>", "<b>Count</b>", "<b>% of Births</b>"],
                fill_color=color,
                align="left",
                font=dict(color="white", size=12)
            ),
            cells=dict(
                values=[
                    list(range(1, top_n + 1)),
                    data_df.get_column("name").to_list(),
                    [f"{x:,}" for x in data_df.get_column("total_count").to_list()],
                    [f"{x:.2f}%" for x in data_df.get_column("pct_of_births").to_list()]
                ],
                fill_color=[["white", "lightgray"] * (top_n // 2 + 1)][:top_n],
                align="left",
                font=dict(size=11)
            )
        )
    
    # Create figure with two tables side by side
    left_key = f"{default_left_decade}s_{default_left_sex}"
    right_key = f"{default_right_decade}s_{default_right_sex}"
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"{default_left_decade}s - {'Male' if default_left_sex == 'M' else 'Female'}",
            f"{default_right_decade}s - {'Male' if default_right_sex == 'M' else 'Female'}"
        ),
        specs=[[{"type": "table"}, {"type": "table"}]],
        horizontal_spacing=0.1
    )
    
    # Add initial traces
    fig.add_trace(
        create_table_trace(interactive_data[left_key], sex_colors[default_left_sex]),
        row=1, col=1
    )
    
    fig.add_trace(
        create_table_trace(interactive_data[right_key], sex_colors[default_right_sex]),
        row=1, col=2
    )
    
    # Create frames for all decade/sex combinations
    frames = []
    available_decades = sorted([d for d in all_decades if f"{d}s_M" in interactive_data])
    
    for decade1 in available_decades:
        for sex1 in ["M", "F"]:
            for decade2 in available_decades:
                for sex2 in ["M", "F"]:
                    key1 = f"{decade1}s_{sex1}"
                    key2 = f"{decade2}s_{sex2}"
                    
                    if key1 not in interactive_data or key2 not in interactive_data:
                        continue
                    
                    frame = go.Frame(
                        data=[
                            create_table_trace(interactive_data[key1], sex_colors[sex1]),
                            create_table_trace(interactive_data[key2], sex_colors[sex2])
                        ],
                        name=f"{key1}_{key2}",
                        layout=go.Layout(
                            annotations=[
                                dict(
                                    text=f"{decade1}s - {'Male' if sex1 == 'M' else 'Female'}", 
                                    x=0.225, y=1.08, xref="paper", yref="paper", showarrow=False,
                                    font=dict(size=14, color="black"), xanchor="center"
                                ),
                                dict(
                                    text=f"{decade2}s - {'Male' if sex2 == 'M' else 'Female'}", 
                                    x=0.775, y=1.08, xref="paper", yref="paper", showarrow=False,
                                    font=dict(size=14, color="black"), xanchor="center"
                                )
                            ]
                        )
                    )
                    frames.append(frame)
    
    fig.frames = frames
    
    # Create dropdown menus
    decade_options = [f"{d}s" for d in available_decades]
    
    # Left table - Decade selector
    left_decade_buttons = []
    for decade in decade_options:
        left_decade_buttons.append(
            dict(
                label=decade,
                method="animate",
                args=[
                    [f"{decade}_{default_left_sex}_{default_right_decade}s_{default_left_sex}"],
                    {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}
                ]
            )
        )
    
    # Right table - Decade selector
    right_decade_buttons = []
    for decade in decade_options:
        right_decade_buttons.append(
            dict(
                label=decade,
                method="animate",
                args=[
                    [f"{default_left_decade}s_{default_left_sex}_{decade}_{default_left_sex}"],
                    {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}
                ]
            )
        )
    
    # Single sex filter that controls both tables
    sex_buttons = [
        dict(
            label="Male",
            method="animate",
            args=[
                [f"{default_left_decade}s_M_{default_right_decade}s_M"],
                {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}
            ]
        ),
        dict(
            label="Female",
            method="animate",
            args=[
                [f"{default_left_decade}s_F_{default_right_decade}s_F"],
                {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}
            ]
        )
    ]
    
    # Find default active indices
    default_left_decade_idx = decade_options.index(f"{default_left_decade}s") if f"{default_left_decade}s" in decade_options else 0
    default_right_decade_idx = decade_options.index(f"{default_right_decade}s") if f"{default_right_decade}s" in decade_options else len(decade_options) - 1
    
    # Update layout with all controls
    fig.update_layout(
        title="Interactive Top Names Comparison",
        height=height,
        showlegend=False,
        updatemenus=[
            # Left table - Decade selector
            dict(
                buttons=left_decade_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                active=default_left_decade_idx,
                x=0.02,
                xanchor="left",
                y=1.25,
                yanchor="top",
                bgcolor="lightgray"
            ),
            # Right table - Decade selector
            dict(
                buttons=right_decade_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                active=default_right_decade_idx,
                x=0.52,
                xanchor="left",
                y=1.25,
                yanchor="top",
                bgcolor="lightgray"
            ),
            # Single sex selector (controls both tables)
            dict(
                buttons=sex_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                active=0 if default_left_sex == "M" else 1,
                x=0.98,
                xanchor="right",
                y=1.25,
                yanchor="top",
                bgcolor="lightgray"
            )
        ]
    )
    
    return fig


def plot_animated_top_names_race(
    df: pl.DataFrame,
    top_n: int = 10,
    sex_colors: dict = None,
    height: int = 600,
    frame_duration: int = 150,
    transition_duration: int = 50
) -> go.Figure:
    """
    Create an animated bar chart race showing the top N names by year with sex filter.
    
    Args:
        df (pl.DataFrame): DataFrame with columns: name, sex, count, year
        top_n (int): Number of top names to display. Defaults to 10.
        sex_colors (dict): Optional dict mapping sex to colors. Defaults to {"M": "#1f77b4", "F": "#ff7f0e"}
        height (int): Height of the figure in pixels. Defaults to 600.
        frame_duration (int): Duration of each frame in milliseconds. Defaults to 150.
        transition_duration (int): Duration of transition between frames in milliseconds. Defaults to 50.
    
    Returns:
        go.Figure: Plotly figure object with animated bar chart
    """
    if sex_colors is None:
        sex_colors = {"M": "#1f77b4", "F": "#ff7f0e"}
    
    # Get top N names for each sex
    male_data = get_top_names_by_year(
        df.filter(pl.col("sex") == "M"), 
        rank=1, 
        top_n=top_n
    )
    female_data = get_top_names_by_year(
        df.filter(pl.col("sex") == "F"), 
        rank=1, 
        top_n=top_n
    )
    
    # Add sex labels
    male_data = male_data.with_columns(pl.lit("Male").alias("sex_label"))
    female_data = female_data.with_columns(pl.lit("Female").alias("sex_label"))
    
    # Combine data
    combined_data = pl.concat([male_data, female_data])
    years = sorted(combined_data.get_column("year").unique())
    
    # Create initial figure with male data
    initial_data = male_data.filter(pl.col("year") == years[0]).sort("rank")
    
    fig = go.Figure(
        data=[go.Bar(
            x=initial_data.get_column("count"),
            y=initial_data.get_column("name"),
            text=initial_data.get_column("name"),
            orientation="h",
            marker=dict(color=sex_colors["M"]),
            textposition="inside",
            textfont=dict(color="white", size=14),
            hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>"
        )]
    )
    
    # Create frames for both sexes
    all_frames = []
    
    for sex_label in ["Male", "Female"]:
        sex_data = combined_data.filter(pl.col("sex_label") == sex_label)
        sex_code = "M" if sex_label == "Male" else "F"
        color = sex_colors[sex_code]
        
        for year in years:
            year_data = sex_data.filter(pl.col("year") == year).sort("rank")
            
            all_frames.append(go.Frame(
                data=[go.Bar(
                    x=year_data.get_column("count"),
                    y=year_data.get_column("name"),
                    text=year_data.get_column("name"),
                    orientation="h",
                    marker=dict(color=color),
                    textposition="inside",
                    textfont=dict(color="white", size=14),
                    hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>"
                )],
                name=f"{sex_label.lower()}_year{year}",
                layout={"sliders": [{"active": years.index(year)}]}
            ))
    
    fig.frames = all_frames
    
    # Create sex filter dropdown buttons
    sex_buttons = []
    for sex_label in ["Male", "Female"]:
        button = dict(
            label=sex_label,
            method="animate",
            args=[
                [f"{sex_label.lower()}_year{years[0]}"],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0}
                }
            ]
        )
        sex_buttons.append(button)
    
    # Create slider steps that reference both male and female frames
    slider_steps = []
    for i, year in enumerate(years):
        slider_steps.append({
            "args": [
                [f"male_year{year}"],
                {
                    "frame": {"duration": frame_duration, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": transition_duration}
                }
            ],
            "method": "animate",
            "label": str(year)
        })
    
    # Update layout
    fig.update_layout(
        updatemenus=[
            # Play/Pause buttons
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": frame_duration, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": transition_duration},
                            "mode": "immediate"
                        }]
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]
                    )
                ],
                direction="left",
                pad={"r": 10, "t": 10},
                showactive=False,
                x=0.02,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
            # Sex dropdown on the right
            dict(
                type="dropdown",
                buttons=sex_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                active=0,
                x=0.98,
                xanchor="right",
                y=1.15,
                yanchor="top"
            )
        ],
        title=f"Top {top_n} Most Popular Baby Names by Year",
        xaxis_title="Number of Babies",
        yaxis_title="",
        xaxis=dict(range=[0, combined_data.get_column("count").max() * 1.1]),
        yaxis=dict(autorange="reversed"),  # Rank 1 at top
        showlegend=False,
        height=height,
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "y": -0.05,
            "xanchor": "left",
            "currentvalue": {
                "prefix": "Year: ",
                "visible": True,
                "xanchor": "right"
            },
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "steps": slider_steps
        }]
    )
    
    return fig