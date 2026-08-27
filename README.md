# Baby Names Shiny Dashboard

An interactive Shiny application for exploring SSA baby names data (1880-2025).

## Features

- **Year Selector**: Filter data by any year from 1880-2025
- **Top Names Cards**: Display most popular male and female names with key metrics:
  - Total count
  - Percentage of births
  - Number of states where the name is popular
- **Interactive Charts**:
  - Top 10 names bar chart
  - State distribution choropleth map
  - Historical trend lines for top names
  - Name diversity over time

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure data files are in place:
   - `data/baby_names.parquet` (national data)
   - `data/baby_names_state.parquet` (state data)
   - `utilities/file_paths.yaml` (configuration)

## Running the App

```bash
shiny run app.py
```

The app will open in your browser at http://localhost:8000

## Technology Stack

- **Shiny for Python**: Web framework
- **DuckDB**: Fast analytical queries on Parquet files
- **Polars**: Data manipulation
- **Plotly**: Interactive visualizations
