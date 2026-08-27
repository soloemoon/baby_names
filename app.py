from shiny import App, ui, render, reactive
import plotly.graph_objects as go
import plotly.express as px
import duckdb
import polars as pl
from pathlib import Path

# Load data paths
import yaml
file_paths = yaml.safe_load(open("./utilities/file_paths.yaml", "r"))

# Initialize DuckDB connection
con = duckdb.connect(database=":memory:")

# Register Parquet files with DuckDB
con.execute(f"""
    CREATE VIEW df_nat AS 
    SELECT * FROM read_parquet('{file_paths["parquet_year_path"]}')
""")

con.execute(f"""
    CREATE VIEW df_state AS 
    SELECT * FROM read_parquet('{file_paths["parquet_state_path"]}')
""")

# Get available years
years = con.execute("SELECT DISTINCT year FROM df_nat ORDER BY year").fetchdf()['year'].tolist()

# UI
app_ui = ui.page_fluid(
    ui.panel_title("Baby Names Dashboard"),
    
    # Year selector
    ui.row(
        ui.column(
            4,
            ui.input_select(
                "year",
                "Select Year:",
                choices={"all": "All Years", **{str(year): str(year) for year in years}},
                selected=str(max(years))
            )
        )
    ),
    
    ui.br(),
    
    # Row 1: Value boxes for top male and female names
    ui.row(
        ui.column(
            6,
            ui.card(
                ui.card_header("Most Popular Male Name"),
                ui.output_ui("male_name_card")
            )
        ),
        ui.column(
            6,
            ui.card(
                ui.card_header("Most Popular Female Name"),
                ui.output_ui("female_name_card")
            )
        )
    ),
    
    ui.br(),
    
    # Row 2: First set of charts
    ui.row(
        ui.column(
            6,
            ui.card(
                ui.card_header("Top 10 Names by Count"),
                ui.output_ui("top_names_chart")
            )
        ),
        ui.column(
            6,
            ui.card(
                ui.card_header("Name Distribution by State"),
                ui.output_ui("state_distribution_chart")
            )
        )
    ),
    
    ui.br(),
    
    # Row 3: Second set of charts
    ui.row(
        ui.column(
            6,
            ui.card(
                ui.card_header("Historical Trend: Top Names"),
                ui.output_ui("trend_chart")
            )
        ),
        ui.column(
            6,
            ui.card(
                ui.card_header("Name Diversity Over Time"),
                ui.output_ui("diversity_chart")
            )
        )
    )
)

# Server
def server(input, output, session):
    
    @reactive.Calc
    def filtered_data():
        """Get filtered data for selected year or all years"""
        year_input = input.year()
        
        if year_input == "all":
            # Aggregate across all years
            nat_query = """
                SELECT name, sex, SUM(count) as count
                FROM df_nat
                GROUP BY name, sex
                ORDER BY count DESC
            """
            nat_df = pl.from_pandas(con.execute(nat_query).fetchdf())
            
            state_query = """
                SELECT state, name, sex, SUM(count) as count
                FROM df_state
                GROUP BY state, name, sex
            """
            state_df = pl.from_pandas(con.execute(state_query).fetchdf())
        else:
            # Query for specific year
            year = int(year_input)
            nat_query = f"""
                SELECT name, sex, count, year
                FROM df_nat
                WHERE year = {year}
                ORDER BY count DESC
            """
            nat_df = pl.from_pandas(con.execute(nat_query).fetchdf())
            
            state_query = f"""
                SELECT state, name, sex, count, year
                FROM df_state
                WHERE year = {year}
            """
            state_df = pl.from_pandas(con.execute(state_query).fetchdf())
        
        return {"nat": nat_df, "state": state_df, "year_input": year_input}
    
    @reactive.Calc
    def top_names():
        """Get top male and female names"""
        data = filtered_data()
        nat_df = data["nat"]
        
        top_male = nat_df.filter(pl.col("sex") == "M").head(1)
        top_female = nat_df.filter(pl.col("sex") == "F").head(1)
        
        return {"male": top_male, "female": top_female}
    
    @output
    @render.ui
    def male_name_card():
        """Display top male name with metrics"""
        top = top_names()["male"]
        if len(top) == 0:
            return ui.p("No data available")
        
        name = top.get_column("name")[0]
        count = top.get_column("count")[0]
        
        # Get additional metrics
        data = filtered_data()
        state_df = data["state"]
        
        # Count states where this name is the top male name
        top_male_by_state = (
            state_df.filter(pl.col("sex") == "M")
            .group_by("state")
            .agg(pl.all().sort_by("count", descending=True).first())
        )
        states_count = len(top_male_by_state.filter(pl.col("name") == name))
        
        # Calculate percentage of total male names
        total_male = data["nat"].filter(pl.col("sex") == "M").get_column("count").sum()
        percentage = (count / total_male * 100) if total_male > 0 else 0
        
        # Count years this name was #1 nationally for males
        years_at_top_query = f"""
            WITH ranked AS (
                SELECT year, name, count,
                       ROW_NUMBER() OVER (PARTITION BY year ORDER BY count DESC) as rank
                FROM df_nat
                WHERE sex = 'M'
            )
            SELECT COUNT(*) as years_count
            FROM ranked
            WHERE name = '{name}' AND rank = 1
        """
        years_at_top = con.execute(years_at_top_query).fetchone()[0]
        
        return ui.div(
            ui.h2(name, style="color: #1f77b4; margin-bottom: 10px;"),
            ui.h4(f"{count:,} babies", style="margin: 5px 0;"),
            ui.p(f"📊 {percentage:.2f}% of {total_male:,.0f} male births", style="margin: 5px 0;"),
            ui.p(f"🗺️ #1 in {states_count} states", style="margin: 5px 0;"),
            ui.p(f"🏆 #1 nationally in {years_at_top} years", style="margin: 5px 0;")
        )
    
    @output
    @render.ui
    def female_name_card():
        """Display top female name with metrics"""
        top = top_names()["female"]
        if len(top) == 0:
            return ui.p("No data available")
        
        name = top.get_column("name")[0]
        count = top.get_column("count")[0]
        
        # Get additional metrics
        data = filtered_data()
        state_df = data["state"]
        
        # Count states where this name is the top female name
        top_female_by_state = (
            state_df.filter(pl.col("sex") == "F")
            .group_by("state")
            .agg(pl.all().sort_by("count", descending=True).first())
        )
        states_count = len(top_female_by_state.filter(pl.col("name") == name))
        
        # Calculate percentage of total female names
        total_female = data["nat"].filter(pl.col("sex") == "F").get_column("count").sum()
        percentage = (count / total_female * 100) if total_female > 0 else 0
        
        # Count years this name was #1 nationally for females
        years_at_top_query = f"""
            WITH ranked AS (
                SELECT year, name, count,
                       ROW_NUMBER() OVER (PARTITION BY year ORDER BY count DESC) as rank
                FROM df_nat
                WHERE sex = 'F'
            )
            SELECT COUNT(*) as years_count
            FROM ranked
            WHERE name = '{name}' AND rank = 1
        """
        years_at_top = con.execute(years_at_top_query).fetchone()[0]
        
        return ui.div(
            ui.h2(name, style="color: #ff7f0e; margin-bottom: 10px;"),
            ui.h4(f"{count:,} babies", style="margin: 5px 0;"),
            ui.p(f"📊 {percentage:.2f}% of {total_female:,.0f} female births", style="margin: 5px 0;"),
            ui.p(f"🗺️ #1 in {states_count} states", style="margin: 5px 0;"),
            ui.p(f"🏆 #1 nationally in {years_at_top} years", style="margin: 5px 0;")
        )
    
    @output
    @render.ui
    def top_names_chart():
        """Bar chart of top 10 names"""
        data = filtered_data()
        nat_df = data["nat"]
        year_label = "All Years" if data["year_input"] == "all" else data["year_input"]
        
        # Get top 10 for each sex
        top_10 = pl.concat([
            nat_df.filter(pl.col("sex") == "M").head(10),
            nat_df.filter(pl.col("sex") == "F").head(10)
        ])
        
        fig = px.bar(
            top_10,
            x="count",
            y="name",
            color="sex",
            orientation="h",
            labels={"count": "Number of Babies", "name": "Name", "sex": "Sex"},
            color_discrete_map={"M": "#1f77b4", "F": "#ff7f0e"},
            height=500
        )
        
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        
        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    
    @output
    @render.ui
    def state_distribution_chart():
        """Choropleth map showing top name by state"""
        data = filtered_data()
        state_df = data["state"]
        year_label = "All Years" if data["year_input"] == "all" else data["year_input"]
        
        # Get top name per state (combined M/F)
        top_by_state = (
            state_df
            .group_by("state")
            .agg(pl.all().sort_by("count", descending=True).first())
        )
        
        fig = go.Figure(data=go.Choropleth(
            locations=top_by_state.get_column("state"),
            z=top_by_state.get_column("count"),
            locationmode="USA-states",
            colorscale="Blues",
            text=top_by_state.get_column("name"),
            hovertemplate="<b>%{location}</b><br>Top Name: %{text}<br>Count: %{z:,}<extra></extra>"
        ))
        
        fig.update_layout(
            geo_scope="usa",
            height=500
        )
        
        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    
    @output
    @render.ui
    def trend_chart():
        """Line chart showing historical trend for current top names"""
        data = filtered_data()
        top = top_names()
        male_name = top["male"].get_column("name")[0] if len(top["male"]) > 0 else None
        female_name = top["female"].get_column("name")[0] if len(top["female"]) > 0 else None
        
        # Query historical data for these names
        query = f"""
            SELECT year, name, sex, SUM(count) as total_count
            FROM df_nat
            WHERE name IN ('{male_name}', '{female_name}')
            GROUP BY year, name, sex
            ORDER BY year
        """
        hist_df = pl.from_pandas(con.execute(query).fetchdf())
        
        fig = px.line(
            hist_df,
            x="year",
            y="total_count",
            color="name",
            labels={"total_count": "Number of Babies", "year": "Year", "name": "Name"},
            height=400
        )
        
        # Add vertical line for selected year (if not "all")
        if data["year_input"] != "all":
            fig.add_vline(x=int(input.year()), line_dash="dash", line_color="gray")
        
        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    
    @output
    @render.ui
    def diversity_chart():
        """Chart showing name diversity over time"""
        data = filtered_data()
        
        query = f"""
            SELECT 
                year,
                sex,
                COUNT(DISTINCT name) as unique_names,
                SUM(count) as total_births
            FROM df_nat
            GROUP BY year, sex
            ORDER BY year
        """
        diversity_df = pl.from_pandas(con.execute(query).fetchdf())
        
        # Calculate concentration (top 10 as % of total)
        fig = px.line(
            diversity_df,
            x="year",
            y="unique_names",
            color="sex",
            labels={"unique_names": "Number of Unique Names", "year": "Year", "sex": "Sex"},
            color_discrete_map={"M": "#1f77b4", "F": "#ff7f0e"},
            height=400
        )
        
        # Add vertical line for selected year (if not "all")
        if data["year_input"] != "all":
            fig.add_vline(x=int(input.year()), line_dash="dash", line_color="gray")
        
        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs="cdn"))

app = App(app_ui, server)
