# -*- coding: utf-8 -*-
from utils import *
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def preprocess_all():

    JSON_DATA_CO2 = "https://nyc3.digitaloceanspaces.com/owid-public/data/co2/owid-co2-data.json"
    CSV_DESCRIPTION_CO2 = 'https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-codebook.csv'
    df_dict_co2, key_description_dict_co2  = download_data(JSON_DATA_CO2, CSV_DESCRIPTION_CO2)

    JSON_DATA_ELEC = "https://nyc3.digitaloceanspaces.com/owid-public/data/energy/owid-energy-data.json"
    CSV_DESCRIPTION_ELEC = 'https://raw.githubusercontent.com/owid/energy-data/refs/heads/master/owid-energy-codebook.csv'
    df_dict_elec, key_description_dict_elec  = download_data(JSON_DATA_ELEC, CSV_DESCRIPTION_ELEC)


    # Find the difference in keys between the two dictionaries
    co2_keys = set(df_dict_co2.keys())
    elec_keys = set(df_dict_elec.keys())

    co2_keys_sorted = sorted(co2_keys)
    elec_keys_sorted = sorted(elec_keys)

    cols = ['year', 'country', 'gdp', 'population', 'gdp_pct_change', 'population_pct_change', 'hdi', 'per_capita_electricity',
            'coal_electricity', 'fossil_electricity', 'gas_electricity', 'hydro_electricity', 'low_carbon_electricity',
            'nuclear_electricity', 'oil_electricity', 'other_renewable_electricity', 'other_renewable_exc_biofuel_electricity',
            'renewables_electricity', 'solar_electricity', 'wind_electricity']

    enormous_super_big_complete_table = pipeline(df_dict_co2, df_dict_elec, cols, columns_to_get = ['hdi'])

    df, melted = get_imf_gdp_projection_for_my_friend_Oscar(encodage = None, url = "https://www.imf.org/-/media/Files/Publications/WEO/WEO-Database/2024/October/WEOOct2024all.ashx")

    enormous_super_big_complete_table_with_gdp_proj_for_Oscar = add_imf_gdp_projections(enormous_super_big_complete_table, melted)

    enormous_super_big_complete_table_with_gdp_proj_for_Oscar = use_imf_data(enormous_super_big_complete_table_with_gdp_proj_for_Oscar, melted)

    # ADD linear population regression
    for c in enormous_super_big_complete_table_with_gdp_proj_for_Oscar.country.unique():
        try:
            enormous_super_big_complete_table_with_gdp_proj_for_Oscar = fill_population_with_regression(enormous_super_big_complete_table_with_gdp_proj_for_Oscar, c)
        except:
            pass

    enormous_super_big_complete_table_with_gdp_proj_for_Oscar['gdp_per_capita'] = np.nan_to_num(
        enormous_super_big_complete_table_with_gdp_proj_for_Oscar['gdp'].astype(float) /
        enormous_super_big_complete_table_with_gdp_proj_for_Oscar['population'].replace(0, np.nan).astype(float)
    )

    enormous_super_big_complete_table_with_gdp_proj_for_Oscar['population_pct_change'] = enormous_super_big_complete_table_with_gdp_proj_for_Oscar['population'].pct_change().astype(float)*100

    enormous_super_big_complete_table_with_gdp_proj_for_Oscar = enormous_super_big_complete_table_with_gdp_proj_for_Oscar.fillna(0)

    # Correct future HDI equal zero
    enormous_super_big_complete_table_with_gdp_proj_for_Oscar.loc[
        enormous_super_big_complete_table_with_gdp_proj_for_Oscar['hdi'] == 0, 'hdi'
    ] = np.nan

    return enormous_super_big_complete_table_with_gdp_proj_for_Oscar, enormous_super_big_complete_table

# Set page config
st.set_page_config(
    page_title="Dynamic Country Development Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Your existing functions (copy from your original code)
def filter_df(data, year, thresholds):
    """Filter dataframe based on year and thresholds"""
    df_year = data[data['year'] == year].copy()
    
    for metric, (threshold, condition) in thresholds.items():
        if metric in df_year.columns:
            if condition == 'bigger':
                df_year = df_year[df_year[metric] > threshold]
            else:  # smaller
                df_year = df_year[df_year[metric] < threshold]
    
    # Create complementary df (countries that don't meet criteria)
    all_countries_year = data[data['year'] == year]
    complementary_df = all_countries_year[~all_countries_year['country'].isin(df_year['country'])]
    
    return df_year, complementary_df

def get_countries_years(enormous_super_big_complete_table, thresholds):
    """Get countries that meet thresholds for each year"""
    years = enormous_super_big_complete_table['year'].unique()
    country_dict = {}

    for year in years:
        df_year, _ = filter_df(enormous_super_big_complete_table, year, thresholds)
        countries = df_year['country'].tolist()
        country_dict[year] = countries

    return country_dict

def calculate_consecutive_years(plot_countries):
    """Calculates consecutive years of strikes per country."""
    result = {}
    for country in plot_countries['country'].unique():
        country_data = plot_countries[plot_countries['country'] == country]
        years = sorted(country_data['year'].unique())

        if not years:
            continue

        consecutive_periods = []
        start_year = years[0]
        current_period = [start_year]

        for i in range(len(years) - 1):
            if years[i+1] == years[i] + 1:
                current_period.append(years[i+1])
            else:
                consecutive_periods.append({
                    'start': start_year, 
                    'end': years[i], 
                    'length': len(current_period)
                })
                start_year = years[i+1]
                current_period = [start_year]

        consecutive_periods.append({
            'start': start_year, 
            'end': years[-1], 
            'length': len(current_period)
        })
        result[country] = consecutive_periods
    return result

def strikes_to_dataframe(strikes):
    """Converts the strikes dictionary to a DataFrame."""
    all_data = []
    for country, periods in strikes.items():
        for period in periods:
            period['country'] = country
            all_data.append(period)
    return pd.DataFrame(all_data)

# Load your data (replace this with your actual data loading)
@st.cache_data
def load_data():
    """
    Replace this function with your actual data loading
    For now, it creates sample data with the same structure
    """
    df_complet, df_orig = preprocess_all()
    return df_complet

def create_dynamic_plot(highlight_country, thresholds, data, from_year, SEQ = 4):
    """Create the main interactive plot"""
    
    # Get countries that meet thresholds for each year
    country_dict = get_countries_years(data, thresholds)
    
    # Prepare plot data
    plot_data = []
    for year, countries in country_dict.items():
        if year >= from_year:
            for country in countries:
                plot_data.append({"year": year, "country": country})
    
    if not plot_data:
        return go.Figure().add_annotation(
            text="No data meets the current criteria",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font_size=16
        )
    
    plot_df = pd.DataFrame(plot_data)
    
    # Calculate consecutive years
    strikes = calculate_consecutive_years(plot_df)
    
    # Create figure
    fig = go.Figure()
    
    # Add all countries as blue dots
    fig.add_trace(go.Scatter(
        x=plot_df['year'],
        y=plot_df['country'],
        mode='markers',
        marker=dict(size=8, color='rgba(70, 130, 180, 0.7)', 
                   line=dict(color='rgba(70, 130, 180, 1)', width=1)),
        name='Tout les pays',
        hovertemplate='%{y}<br>Année: %{x}<extra></extra>'
    ))
    
    # Highlight specific country
    highlight_data = plot_df[plot_df['country'] == highlight_country]
    if not highlight_data.empty:
        # Get streak information
        country_strikes = strikes.get(highlight_country, [])
        if country_strikes:
            longest_streak = max(country_strikes, key=lambda x: x['length'])
            streak_info = f"Années consecutives: {longest_streak['start']}-{longest_streak['end']} ({longest_streak['length']} years)"
        else:
            streak_info = "No consecutive years"
        
        fig.add_trace(go.Scatter(
            x=highlight_data['year'],
            y=highlight_data['country'],
            mode='lines+markers',
            line=dict(color='red', width=4),
            marker=dict(size=12, color='red', symbol='diamond'),
            name=f'{highlight_country}<br>{streak_info}',
            hovertemplate='%{y}<br>Year: %{x}<br><b>Highlighted</b><extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Pays dans les critères (Depuis: {from_year})',
        xaxis_title='Année',
        yaxis_title='Pays',
        height=700,
        hovermode='closest',
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1)
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)')

    strikes_df = strikes_to_dataframe(strikes)
    # strikes_df[strikes_df['country'] == highlight]
    conclusion = strikes_df[(strikes_df['start'] >= from_year) | ((strikes_df['end'] >= from_year) & (strikes_df['start'] <= from_year))]
    
    
    return fig, get_current_PVD(conclusion, strike_size = SEQ)

def create_summary_stats(data, thresholds, from_year):
    """Create summary statistics"""
    country_dict = get_countries_years(data, thresholds)
    
    # Filter by from_year
    filtered_dict = {year: countries for year, countries in country_dict.items() if year >= from_year}
    
    # Calculate statistics
    total_country_years = sum(len(countries) for countries in filtered_dict.values())
    unique_countries = len(set(country for countries in filtered_dict.values() for country in countries))
    years_with_data = len([year for year, countries in filtered_dict.items() if countries])
    
    return {
        'total_country_years': total_country_years,
        'unique_countries': unique_countries,
        'years_with_data': years_with_data,
        'avg_countries_per_year': total_country_years / max(len(filtered_dict), 1)
    }

# Application principale Streamlit
def rodar_esse_negocio():
    st.title("Analyse Dynamique des Pays")
    st.markdown("Visualisation interactive des pays atteignant les seuils de développement au fil du temps")
    
    # Chargement des données
    with st.spinner("Chargement des données..."):
        data = load_data()
    
    # Contrôles dans la barre latérale
    st.sidebar.header("📊 Contrôles")
    
    # Sélection de l’année de départ
    from_year = st.sidebar.selectbox(
        "Année DEPUIS :",
        options=list(range(2000, 2019)),
        index=15  # Valeur par défaut : 2015
    )

    # Sélection d'années de consecutives
    min_conscutiver_years = st.sidebar.selectbox(
        "Années Consecutives :",
        options=list(range(1, 18)),
        index=3  # Valeur par défaut : 4
    )
    
    # Sélection du pays à mettre en évidence
    available_countries = sorted(data['country'].unique())
    highlight_country = st.sidebar.selectbox(
        "Mettre en évidence le pays :",
        options=available_countries,
        index=available_countries.index('Senegal') if 'Senegal' in available_countries else 0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎚️ Seuils")
    
    # Curseurs de seuils
    gdp_threshold = st.sidebar.slider(
        "Croissance du PIB (%) - Minimum :",
        min_value=-1.0, max_value=5.0, value=1.5, step=0.1,
        help="Les pays doivent avoir une croissance du PIB SUPÉRIEURE à ce seuil"
    )
    
    hdi_threshold = st.sidebar.slider(
        "IDH - Maximum :",
        min_value=0.30, max_value=0.90, value=0.70, step=0.02,
        help="Les pays doivent avoir un IDH INFÉRIEUR à ce seuil"
    )
    
    pop_threshold = st.sidebar.slider(
        "Croissance démographique (%) - Minimum :",
        min_value=0.5, max_value=3.0, value=1.1, step=0.1,
        help="Les pays doivent avoir une croissance démographique SUPÉRIEURE à ce seuil"
    )
    
    # Dictionnaire des seuils
    thresholds = {
        'gdp_pct_change': (gdp_threshold, 'bigger'),
        'hdi': (hdi_threshold, 'smaller'),
        'population_pct_change': (pop_threshold, 'bigger'),
    }
    
    # Affichage des seuils actuels
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Critères actuels :**")
    st.sidebar.markdown(f"• Croissance du PIB > {gdp_threshold}%")
    st.sidebar.markdown(f"• IDH < {hdi_threshold:.2f}")
    st.sidebar.markdown(f"• Croissance démographique > {pop_threshold}%")
    
    # Zone de contenu principal
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Générer et afficher le graphique principal
        with st.spinner("Génération du graphique..."):
            fig, PVDs = create_dynamic_plot(highlight_country, thresholds, data, from_year, SEQ = min_conscutiver_years)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Statistiques résumées
        st.subheader("📈 Statistiques Résumées")
        stats = create_summary_stats(data, thresholds, from_year)
        
        st.metric("Pays uniques", stats['unique_countries'])
        st.metric("Total pays-années", stats['total_country_years'])
        st.metric("Années avec données", stats['years_with_data'])
        st.metric("Moy. pays/année", f"{stats['avg_countries_per_year']:.1f}")
        
        # Informations sur le pays mis en évidence
        st.subheader(f"🎯 {highlight_country}")
        
        country_dict = get_countries_years(data, thresholds)
        highlight_years = [year for year, countries in country_dict.items() 
                          if highlight_country in countries and year >= from_year]
        
        if highlight_years:
            st.success(f"Qualifié pour {len(highlight_years)} années")
            st.write(f"Années : {min(highlight_years)}-{max(highlight_years)}")
            
            plot_data = [{"year": year, "country": highlight_country} for year in highlight_years]
            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                strikes = calculate_consecutive_years(plot_df)
                
                if highlight_country in strikes:
                    longest = max(strikes[highlight_country], key=lambda x: x['length'])
                    st.info(f"Période la plus longue : {longest['length']} années ({longest['start']}-{longest['end']})")
        else:
            st.warning("Aucune année qualifiée trouvée")
    
    # Tableau de données (déroulable)
    with st.expander("📋 Voir les données filtrées", expanded=False):
        country_dict = get_countries_years(data, thresholds)
        current_data = []
        for year, countries in country_dict.items():
            if year >= from_year:
                for country in countries:
                    country_row = data[(data['country'] == country) & (data['year'] == year)].iloc[0]
                    current_data.append(country_row.to_dict())
        
        if current_data:
            df_display = pd.DataFrame(current_data)
            st.markdown(f"## Données d'énergie de [Our World in Data](https://github.com/owid)")
            st.dataframe(df_display, use_container_width=True)
            st.markdown("---")
            st.markdown(f"## Pays qui suivent les critères pendant une séquence d'au moins {min_conscutiver_years} années")
            st.dataframe(PVDs, use_container_width=True)
        else:
            st.info("Aucune donnée ne correspond aux critères actuels")
    
    # Instructions
    with st.expander("ℹ️ Comment utiliser", expanded=False):
        st.markdown("""
        **Contrôles :**
        - **Année DEPUIS** : Année de début de l'analyse
        - **Mettre en évidence le pays** : Pays à afficher en rouge
        - **Seuils** : Ajuster les critères de sélection des pays
        
        **Graphique :**
        - Points bleus : Tous les pays répondant aux critères
        - Ligne rouge : Trajectoire du pays sélectionné
        - Survolez pour les détails
        
        **Critères :**
        Les pays doivent satisfaire à TOUS les critères de seuil **simultanément** pour chaque année.
        """)

if __name__ == "__main__":
    rodar_esse_negocio()
