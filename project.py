import streamlit as st
import json
import plotly.graph_objects as go
import numpy as np  # Add this line to import NumPy
import geopandas as gpd
import ipywidgets as widgets
import networkx as nx
from IPython.display import display, clear_output
import pandas as pd
import plotly.express as px
from PIL import Image, ImageDraw, ImageFont

# Set the page config as the very first Streamlit command
st.set_page_config(page_title="Professor's Dashboard", layout="wide")

# Page Navigation
st.sidebar.title("Choose a Page")
page = st.sidebar.selectbox("", ["Main Page", "Dashboard 1: Professors", "Dashboard 2: Analysis of fields of study", "Interactive graphs", "Creative Visualization"])

# Load the JSON data with caching
@st.cache_data()  # Use caching to load the data only once
def load_data():
    with open('professors_data.json', 'r') as file:
        data = json.load(file)
    return data

data = load_data()
professors_names = list(data.keys())

# Page Navigation
if page == "Main Page":
    st.title("Main Page - Professor Selection and Metrics")

    # Professor selection and information display
    selected_professor = st.selectbox('Select a Professor', professors_names)
    if selected_professor:
        prof_info = data[selected_professor]
        st.write(f"Professor: {selected_professor}")
        st.write(f"H-index: {prof_info.get('Index_H', 'N/A')}")
        st.write(f"Citations: {prof_info.get('Citation_Count', 'N/A')}")  # Corrected key
        st.write(f"Publications: {prof_info.get('Paper_Count', 'N/A')}")  # Corrected key
        st.write(f"Country: {prof_info.get('Country', 'N/A')}")

        # Publication with the most citations
        if 'Papers' in prof_info:
            publications = prof_info['Papers']
            if publications:
                most_cited_publication = max(publications, key=lambda x: x.get('Citation_Count', 0))
                st.write("Publication with Most Citations:")
                st.write(f"Title: {most_cited_publication.get('Title', 'N/A')}")
                st.write(f"Citations: {most_cited_publication.get('Citation_Count', 'N/A')}")
                st.write(f"Year: {most_cited_publication.get('Year_of_Publication', 'N/A')}")

                # Add more info about the top publication if available
                # Add more info about the top publication if available
                authors_details = most_cited_publication.get('Authors_Details', [])
                if authors_details:
                    if isinstance(authors_details, list):
                        authors = ', '.join(author.get('Name', 'N/A') for author in authors_details)
                        if all(author.get('Name') == 'N/A' for author in authors_details):
                            authors = 'N/A'
                    else:
                        authors = ', '.join(authors_details)
                        st.write(f"Authors: {authors}")

                st.write(f"Venue: {most_cited_publication.get('Venue_Name', 'N/A')}")
                st.write(f"URL: {most_cited_publication.get('Paper_URL', 'N/A')}")

        # Top Co-Authors (showing only the top three)
        if 'Co-authors' in prof_info:
            coauthors = prof_info['Co-authors']
            # Sorting co-authors by collaborations, descending, and picking the top three
            top_coauthors = sorted(coauthors.items(), key=lambda item: item[1], reverse=True)[:5]
            st.write("Top Co-Authors:")
            for coauthor, collaborations in top_coauthors:
                st.write(f"Co-author: {coauthor}, Collaborations: {collaborations}")

        # Publication Types Distribution (assuming 'Publication_Types' structure remains the same)
        if 'Publication_Types' in prof_info:
            publication_types = prof_info['Publication_Types']
            st.write("Publication Types Distribution:")
            for pub_type, count in publication_types.items():
                st.write(f"{pub_type}: {count}")

elif page == "Dashboard 1: Professors":
    st.title("Dashboard 1: Professors")

#### Publication counts per professor
    # Detailed graphs for all professors
    publications_counts = [data[name].get('Paper_Count', 0) for name in professors_names]
    h_indices = [data[name].get('Index_H', 0) for name in professors_names]

    sorted_data = sorted(zip(professors_names, publications_counts), key=lambda x: x[1], reverse=True)
    fig_pub = go.Figure(data=[
        go.Bar(
            x=[x[0] for x in sorted_data],
            y=[x[1] for x in sorted_data],
            marker_color='#3569aa',
        )
    ])
    fig_pub.update_layout(
        title='Publication Counts per Professor',
        xaxis_tickangle=-90,
        xaxis_title="Professor",
        yaxis_title="Number of Publications",
        template='plotly_white',
        height=700, 
    )
    fig_pub.update_layout(autosize=True)
    st.plotly_chart(fig_pub, use_container_width=True)

#### H-index distribution
    fig_h_index = go.Figure(data=[
        go.Histogram(x=h_indices, nbinsx=10, marker_color='#3569aa', opacity=0.9)
    ])
    fig_h_index.update_layout(
        title='Histogram of H-indices',
        xaxis_title="H-index",
        yaxis_title="Frequency",
        template='plotly_white',
        bargap=0.1
    )
    fig_h_index.update_layout(autosize=True)
    st.plotly_chart(fig_h_index, use_container_width=True)

    # Extracting year and citation data
    publication_years = []
    citation_counts = []
    for prof_data in data.values():
        for paper in prof_data.get('Papers', []):
            year = paper.get('Year_of_Publication')
            if year:
                publication_years.append(year)
                citation_counts.append(paper.get('Citation_Count', 0))

#### Citation count over years
    # Calculate average citation count per year
    avg_citation_counts = {}
    for year, citation_count in zip(publication_years, citation_counts):
        if year in avg_citation_counts:
            avg_citation_counts[year].append(citation_count)
        else:
            avg_citation_counts[year] = [citation_count]

    years = sorted(avg_citation_counts.keys())
    avg_citations = [np.mean(avg_citation_counts[year]) for year in years]

    fig_citation = go.Figure()

    fig_citation.add_trace(
        go.Scatter(
            x=years,
            y=avg_citations,
            mode='lines+markers',
            marker=dict(color='#3569ab', size=5),
            line=dict(color='#3569aa')
        )
    )

    fig_citation.update_layout(
        title='Citation Impact Over Time',
        xaxis_title='Year',
        yaxis_title='Average Citation Count',
        template='plotly_white'
    )
    fig_citation.update_layout(autosize=True)
    st.plotly_chart(fig_citation, use_container_width=True)

#### Gender Distribution
    # %% md
    ### Gender Distribution of Authors
    # %%
    # Extracting gender counts
    gender_counts = {}
    for prof_data in data.values():
        gender = prof_data.get('Gender')
        if gender:
            if gender in gender_counts:
                gender_counts[gender] += 1
            else:
                gender_counts[gender] = 1

    # Create Plotly pie chart with custom colors
    fig_gender = go.Figure(data=[go.Pie(labels=list(gender_counts.keys()),
                                        values=list(gender_counts.values()),
                                        hole=.3,  # Creates a donut-shaped pie chart
                                        hoverinfo='label+percent',
                                        textinfo='value',
                                        marker=dict(colors=['#ff6666','#3569aa']))])  # Custom colors

    fig_gender.update_layout(
        title='Gender Distribution of Authors',
        template='plotly_white'
    )

    fig_gender.update_layout(autosize=True)
    st.plotly_chart(fig_gender, use_container_width=True)

elif page == "Dashboard 2: Analysis of fields of study":
    st.title("Dashboard 2: Analysis of fields of study")

##### Comparison between temporal analysis of fields growth
    # Publication Trends by Field
    field_publications = {}
    for prof_data in data.values():
        for paper in prof_data.get('Papers', []):
            fields = paper.get('Fields_of_Study', []) if paper.get('Fields_of_Study') is not None else []
            year = paper.get('Year_of_Publication')
            for field in fields:
                if field not in field_publications:
                    field_publications[field] = {year: 1}
                else:
                    if year in field_publications[field]:
                        field_publications[field][year] += 1
                    else:
                        field_publications[field][year] = 1

    fields_growth_rate = {}
    max_years = max(len(publications) for publications in field_publications.values())
    for field, publications in field_publications.items():
        years = sorted(publications.keys())
        counts = [publications[year] for year in years]
        growth_rate = [(counts[i] - counts[i - 1]) / counts[i - 1] * 100 if i > 0 else 0 for i in range(len(counts))]
        growth_rate += [0] * (max_years - len(growth_rate))
        fields_growth_rate[field] = growth_rate[:max_years]

    # Initialize the figure and output widget
    fig = go.Figure()

    # Function to update the plot based on dropdown selections
    def update_plot(field1, field2):
        # Initialize a new figure to start fresh
        fig = go.Figure()

        years = list(range(1, max_years + 1))
        if field1:  # Add the first field if selected
            fig.add_trace(go.Scatter(x=years, y=fields_growth_rate[field1],
                                     mode='lines+markers', name=field1,
                                     line=dict(color='#3569aa')))
        if field2:  # Add the second field if selected
            fig.add_trace(go.Scatter(x=years, y=fields_growth_rate[field2],
                                     mode='lines+markers', name=field2,
                                     line=dict(color='lightblue')))

        # Update layout only if at least one field is selected
        if field1 or field2:
            fig.update_layout(
                title='Temporal Analysis of Field Growth',
                xaxis_title='Year',
                yaxis_title='Growth Rate (%)',
                legend_title='Field',
                template='plotly_white'
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightBlue')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightBlue')

        fig.update_layout(autosize=True)
        st.plotly_chart(fig, use_container_width=True)


    # Dropdown widgets for field selection
    dropdown1 = st.selectbox('Field 1', [None] + list(fields_growth_rate.keys()))
    dropdown2 = st.selectbox('Field 2', [None] + list(fields_growth_rate.keys()))

    # Arrange the dropdowns and the plot output in a vertical layout
    update_plot(dropdown1, dropdown2)

#### Citation count for each field of study
    # Comparison of Citation Counts Across Fields
    field_citation_counts = {}
    for field, publications in field_publications.items():
        counts = sum(publications.values())
        field_citation_counts[field] = counts

    # Sorting the fields by citation counts in descending order
    sorted_fields = sorted(field_citation_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_field_names = [field[0] for field in sorted_fields]
    sorted_citation_counts = [field[1] for field in sorted_fields]

    # Creating the bar chart
    fig = go.Figure(data=go.Bar(x=sorted_field_names, y=sorted_citation_counts,
                                marker_color='#3569aa'))  # Set the color of the bars

    # Enhancing the chart appearance
    fig.update_layout(title='Comparison of Citation Counts Across Fields',
                    xaxis_tickangle=-45,
                    xaxis_title='Field of Study',
                    yaxis_title='Total Citation Counts',
                    template='plotly_white')

    # Show the plot
    fig.update_layout(autosize=True)
    st.plotly_chart(fig, use_container_width=True)

#### Publication Trends by Field
    # Plotting widget output for Publication Trends by Field
    plot_output = st.empty()

    # Plotting widget output
    plot_output = st.empty()


    def update_plot(selected_field):
        plot_output.empty()  # Clear the current output
        fig = go.Figure()

        # Add data for selected field if any
        if selected_field:
            years = sorted(field_publications[selected_field].keys())
            counts = [field_publications[selected_field][year] for year in years]
            fig.add_trace(go.Scatter(x=years, y=counts, mode='lines+markers', name=selected_field, line=dict(color='#3569aa')))

        # Update plot layout
        fig.update_layout(
            title='Publication Trends by Field',
            xaxis_title='Year',
            yaxis_title='Number of Publications',
            legend_title='Field',
            template='plotly_white'
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightBlue')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightBlue')

        fig.update_layout(autosize=True)
        st.plotly_chart(fig, use_container_width=True)


    # Dropdown widget for field selection
    dropdown = st.selectbox('Select a Field:', [None] + list(field_publications.keys()))

    # Call update_plot function when dropdown selection changes
    update_plot(dropdown)

elif page == "Interactive graphs":
    st.title("Interactive graphs")

#### Ego-Network
    # Dropdown widget
    selected_prof = st.selectbox("Select a Professor", list(data.keys()))

    G = nx.Graph()
    professor_data = data.get(selected_prof, {})
    co_authors = professor_data.get('Co-authors', {})

    # Populate your graph with edges
    for co_author in co_authors.keys():
        G.add_edge(selected_prof, co_author)

    # Position nodes using one of the layout options in NetworkX
    pos = nx.spring_layout(G)

    fig = go.FigureWidget()

    with fig.batch_update():
        fig.data = []  # Clear existing data

        # Extract node positions for plotting
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])  # line breaks
            edge_y.extend([y0, y1, None])  # line breaks

        # Create edge traces
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'))

        # Create node traces
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                color=[len(G.adj[node]) for node in G.nodes()],
                size=10,
                line=dict(width=2)
            ),
            text=node
        ))

        # Update layout
        fig.update_layout(
            title='<br>Network graph of co-authorships',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="This graph represents the co-author network of the selected professor.",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002 ) ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )

        # Set text for nodes
        for node in G.nodes():
            fig.data[1].text = list(G.nodes())

    # Display the network graph
    fig.update_layout(autosize=True)
    st.plotly_chart(fig, use_container_width=True)

#### Treemap graph
    rows = []

    for professor_name, details in data.items():
        total_citations = details.get("Citation_Count", 0)
        paper_citations_sum = 0

        if "Papers" in details:
            for paper in details["Papers"]:
                paper_title = paper["Title"]
                paper_citations = paper.get("Citation_Count", 0)
                rows.append({
                    "labels": paper_title,
                    "parents": professor_name,
                    "values": paper_citations,
                })
                paper_citations_sum += paper_citations

        remaining_citations = total_citations - paper_citations_sum
        
        if remaining_citations > 0:
            rows.append({
            "labels": professor_name,
            "parents": "",
            "values": remaining_citations,
        })


    df = pd.DataFrame(rows)
    df['values'].fillna(0, inplace=True)
    df = df[df['values'] > 0]
    # Load a built-in color scale
    original_scale = px.colors.sequential.Blues

    # Cut the first 30% of the colors
    cut_point = int(len(original_scale) * 0.4)
    new_scale = original_scale[cut_point:]

    # Create a custom continuous color scale
    custom_scale = [(i / (len(new_scale) - 1), color) for i, color in enumerate(new_scale)]

    # Create the treemap using the custom color scale
    fig_treemap = px.treemap(df, path=['parents', 'labels'], values='values',
                            color='values', hover_data=['labels'],
                            color_continuous_scale=custom_scale,
                            title='Interactive Treemap of Citation Distribution')

    fig_treemap.update_layout(autosize=False, coloraxis_colorbar_title= "Citations",  
                              font=dict(family="Helvetica, Arial, sans-serif", 
                                        size=20, color="white"), 
                                        width = 1700, height = 800)

    fig_treemap.update_traces(
    textfont=dict(family="Arial, sans-serif", size=20, color="white"),
    textposition='middle center',
    hoverinfo='label+value+name',
    texttemplate="<b>%{label}</b><br>%{value}"
    )


    st.plotly_chart(fig_treemap, use_container_width=True)

#### Geographical data
    professors_country = [{'Name': key, 'Country': value['Country']} for key, value in data.items()]

    professor_df = pd.DataFrame(professors_country)

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    professor_df['Country'] = professor_df['Country'].replace({
    'USA': 'United States of America',
    })

    merged = world.merge(professor_df, how="right", left_on="name", right_on="Country")

    professor_counts = merged.groupby('iso_a3').size().reset_index(name='Professor_Count')

    df_geo = merged.merge(professor_counts, on='iso_a3', how='left')

    df_geo['latitude'] = None
    df_geo['longitude'] = None 

    df_geo.loc[df_geo['Country'] == "Albania", 'latitude'] = 41.153332
    df_geo.loc[df_geo['Country'] == "Albania", 'longitude'] = 20.168331
    df_geo.loc[df_geo['Country'] == "Greece", 'latitude'] = 39.074208
    df_geo.loc[df_geo['Country'] == "Greece", 'longitude'] = 21.824312
    df_geo.loc[df_geo['Country'] == "Italy", 'latitude'] = 41.871940
    df_geo.loc[df_geo['Country'] == "Italy", 'longitude'] = 12.567380
    df_geo.loc[df_geo['Country'] == "India", 'latitude'] = 20.595164
    df_geo.loc[df_geo['Country'] == "India", 'longitude'] = 78.963060
    df_geo.loc[df_geo['Country'] == "Israel", 'latitude'] = 31.046051
    df_geo.loc[df_geo['Country'] == "Israel", 'longitude'] = 34.851612
    df_geo.loc[df_geo['Country'] == "United States of America", 'latitude'] = 37.090240
    df_geo.loc[df_geo['Country'] == "United States of America", 'longitude'] = -95.712891
    df_geo.loc[df_geo['Country'] == "South Korea", 'latitude'] = 35.907757
    df_geo.loc[df_geo['Country'] == "South Korea", 'longitude'] = 127.766922
    df_geo.loc[df_geo['Country'] == "Germany", 'latitude'] = 51.165691
    df_geo.loc[df_geo['Country'] == "Germany", 'longitude'] = 10.451526
    df_geo.loc[df_geo['Country'] == "Turkey", 'latitude'] = 38.963745
    df_geo.loc[df_geo['Country'] == "Turkey", 'longitude'] = 35.243322
    df_geo.loc[df_geo['Country'] == "United Kingdom", 'latitude'] = 55.378051
    df_geo.loc[df_geo['Country'] == "United Kingdom", 'longitude'] = -3.435973
    df_geo.loc[df_geo['Country'] == "Bulgaria", 'latitude'] = 42.733883
    df_geo.loc[df_geo['Country'] == "Bulgaria", 'longitude'] = 25.485830
    df_geo.loc[df_geo['Country'] == "Netherlands", 'latitude'] = 52.132633
    df_geo.loc[df_geo['Country'] == "Netherlands", 'longitude'] = 5.291266
    df_geo.loc[df_geo['Country'] == "Chile", 'latitude'] = -35.675147
    df_geo.loc[df_geo['Country'] == "Chile", 'longitude'] = -71.542969
    df_geo.loc[df_geo['Country'] == "Iran", 'latitude'] = 32.453814
    df_geo.loc[df_geo['Country'] == "Iran", 'longitude'] = 48.348936


    def create_choropleth(df, selected_country = None):
        fig = go.Figure(data=[go.Choropleth(
        locations=df_geo['iso_a3'],  # Use the ISO A3 country codes
        z=df_geo['Professor_Count'],  # Data to be visualized
        text=df_geo['Country'],  # Hover text
        colorscale='Blues',  # Color scale
        autocolorscale=False,
        marker_line_color= "darkblue",
        marker_line_width=0.5,
        colorbar_title='Number of Professors',
        zmin=0,
        zmax=42,
        colorbar=dict(
            len=0.3,
            lenmode='fraction'
        )
    )])

        fig.update_layout(
        width=3500, 
        height=500,
        title_text='Global Distribution of Professors',
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor='lightblue',
            projection_type='equirectangular'
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )
        return fig

    def adjust_projection_scale(country):
        scales = {
        'India': 6, 'Chile': 3, 'United States of America': 2,
        'Israel': 15, 'Albania': 20, 'Netherlands': 20, 'Germany': 15,
        'Italy': 15, 'Greece': 20,
        'Turkey': 15, 'Iran': 12, 'United Kingdom': 15, 'Bulgaria': 20,
        
    }
        return scales.get(country, 5)  # Default scale
    
    st.title("Global Distribution of Professors")

    with st.container():
        country = st.selectbox('Select a Country:', ['All'] + sorted(df_geo['Country'].unique().tolist()))
        col1, col3, col2 = st.columns([4, 0.3, 2])

    with col1:
        if country == 'All':
            fig = create_choropleth(df_geo)
        else:
            selected_data = df_geo[df_geo['Country'] == country]
            scale = adjust_projection_scale(country)
            lat = selected_data['latitude'].values[0]
            lon = selected_data['longitude'].values[0]
            fig = create_choropleth(selected_data)
            fig.update_geos(center={"lat": lat, "lon": lon}, projection_scale=scale)
        
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        st.empty()
    
    with col2:
        st.subheader("Professors List")
        if country != 'All':
            displayed_professors = professor_df[professor_df['Country'] == country]
            st.dataframe(displayed_professors[['Name']], width=600, hide_index = True)
        else:
            displayed_professors = professor_df
            st.dataframe(displayed_professors[['Name', 'Country']], width=600, hide_index = True)

#### Creative visualization
elif page == "Creative Visualization":
    st.title("Creative Visualization") 

    def draw_overlay(book_img_path):
        # Load the original book image
        book_img = Image.open(book_img_path).convert("RGBA")
        
        # Create masks and corresponding drawings
        masks = [Image.new("L", book_img.size, 0) for _ in range(10)]
        draws = [ImageDraw.Draw(mask) for mask in masks]

        # Define coordinates for the masks
        coordinates = [
            (225, 899, 790, 910),   # Biology
            (225, 867, 790, 899),   # Mathematics
            (225, 828, 790, 867),   # Engineering
            (225, 789, 790, 828),   # Political Science
            (225, 742, 790, 789),   # Sociology
            (225, 687, 790, 742),   # Medicine
            (225, 619, 790, 687),   # Economics
            (225, 546, 790, 619),   # Psychology
            (225, 418, 790, 546),   # Computer Science
            (225, 140, 790, 418)    # Business
        ]
        
        # Draw rectangles on each mask
        for draw, coord in zip(draws, coordinates):
            draw.rectangle(coord, fill=100)

        # Define colors for each overlay
        colors = [
            (0, 255, 0, 128),       # Green
            (0, 0, 255, 128),       # Blue
            (170, 170, 170, 128),   # Grey
            (255, 170, 0, 128),     # Light orange
            (255, 255, 0, 128),     # Yellow
            (255, 0, 0, 128),       # Red
            (0, 170, 255, 128),     # Sky Blue
            (255, 0, 255, 128),     # Magenta
            (0, 255, 255, 128),     # Cyan
            (85, 0, 255, 128)       # Indigo
        ]

        # Apply overlays using their respective masks
        for mask, color in zip(masks, colors):
            overlay = Image.new("RGBA", book_img.size, color)
            book_img.paste(overlay, mask=mask)

        return book_img

    def main():
        st.title("Book Publications Visualization")

        # Hard-code the path to the book image that's already on the server
        book_img_path = 'Creative Visualization/Creative Vis.png'  # Update this path

        # Process the image
        result_img = draw_overlay(book_img_path)
        
        # Save the processed image to a buffer
        from io import BytesIO
        buffer = BytesIO()
        result_img.save(buffer, format="PNG")
        buffer.seek(0)

        # Display the image
        st.image(buffer, caption='Book sectioned according to the amount of publications for each field of study', use_column_width=True)

    if __name__ == "__main__":
        main()