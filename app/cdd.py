import asyncio
import io
import glob
import json

import numpy as np 
import pandas as pd
import shapely
import geopandas as gpd
import xarray as xr
import rioxarray

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import matplotlib.font_manager as font_manager

import plotly.express as px
import plotly.graph_objects as go
from shiny import App, Inputs, Outputs, Session, ui, render, reactive
from shinywidgets import render_plotly, reactive_read, render_widget, output_widget
import ipywidgets

from pathlib import Path

from utils import *

# shiny run --reload drought.py

updating = False

# # Load data
# df = pd.read_csv(
#     "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")
# df.columns = [col.replace("AAPL.", "") for col in df.columns]
# df_dates = [pd.to_datetime(date) for date in sorted(df.Date.values)]

# open historical and forecast data
h = None if updating else xr.open_dataset(Path(__file__).parent /'mnt/data/zarr/h.zarr', engine='zarr', consolidated=True, decode_coords="all", chunks=None,).compute()
f = None if updating else xr.open_dataset(Path(__file__).parent /'mnt/data/zarr/f.zarr', engine='zarr', consolidated=True, decode_coords="all", chunks=None,).compute()

# if not updating:
#     # the data variables can come back in a different order when you read in the Zarr instead of the NetCDF
#     f = f[['5%', '20%', 'perc', '80%', '95%']]

# open country and states boundary layers
countries = gpd.GeoDataFrame(columns=['name', 'geometry']) if updating else gpd.read_parquet(Path(__file__).parent / 'mnt/data/vector/countries.parquet')
states = gpd.GeoDataFrame(columns=['name', 'country', 'geometry']) if updating else gpd.read_parquet(Path(__file__).parent / 'mnt/data/vector/states.parquet')

# this is used in the the filename for downloading plots and tables, but is also used in slider values
min_slider_date = None if updating else '1991-01-01'
max_slider_date = None if updating else pd.to_datetime(h.time.values[-5]).strftime('%Y-%m-%d')
forecast_date = None if updating else pd.to_datetime(sorted(f.time.values)[0]).strftime('%Y-%m-%d')
dates = None if updating else [pd.to_datetime(date).strftime('%Y-%m-%d') for date in sorted(h.time.values[:-4])]

min_index = None if updating else 0
max_year = None if updating else pd.to_datetime(h.time.values[-1]).year
skip_index = None if updating else dates.index(f'{max_year - 4}-01-01')
max_index = None if updating else len(dates) - 1

# point the app to the static files directory
static_dir = Path(__file__).parent / "www"
# get the font based on the path
ginto = font_manager.FontProperties(fname='./www/GintoNormal-Regular.ttf')
ginto_medium = font_manager.FontProperties(fname='./www/GintoNormal-Medium.ttf')

app_ui = ui.page_fluid(
    # css
     ui.tags.head(
        ui.include_css(static_dir / 'stylesheet.css'),        
        ui.include_js('./scripts/reset-sidebar-visibility.js', method='inline'),
        ui.include_js('./scripts/sidebar-visibility.js', method='inline'),
    ),

    ui.div({'id': 'layout'},

        ui.div({'id': 'navbar'},
            ui.div({'id': 'logo-container'}, 
                ui.div({'id': 'logo-inner-container'},
                    ui.img(src='woodwell-risk.png', width='45px', alt='Woodwell Climate Research Center Risk group logo'),
                    ui.p({'id': 'org-title'}, 'Woodwell Risk'),
                ),
            ),
            ui.div({'id': 'menu-container'},
                ui.div({'id': 'menu-inner-container'},
                ui.input_action_button('about_button', 'About',),
                ui.input_action_button('settings_button', 'Settings'),
                ),
            ),
        ),

        # wrapper container for sidebar and main panel
        ui.div({'id': 'container'},
            # sidebar
            ui.div({'id': 'sidebar-container', 'class': 'show'},
                ui.div({'id': 'sidebar'}, 
                    ui.div({'id': 'sidebar-inner-container'},

                        ui.div({'class': 'select-label-container'},
                            ui.p({'class': 'select-label'}, 'Select a country:')
                        ),
                        ui.input_text("country_filter", label='', placeholder='Filter by name'),
                        ui.input_select('country_select', '', [], size=5),

                        ui.div({'class': 'select-label-container'},
                            ui.p({'class': 'select-label'}, 'Select a state:')
                        ),
                        ui.input_select('state_select', '', [], size=5),

                        ui.div({'id': 'process-data-container'},
                            ui.input_task_button("process_data_button", label="Run"),
                        ),
                    )
                ),
            ),

            # figures and tables
            ui.div({"id": "main-container"},
                ui.div({'id': 'main'},
                    ui.navset_tab(

                        # timeseries and table tab
                        ui.nav_panel('Timeseries', 

                            ui.div({'id': 'download-timeseries-container', 'class': 'download-container'},
                                ui.download_link("download_timeseries_link", 'Download timeseries')
                            ),
                            ui.div({'id': 'timeseries-container'},
                                ui.div({'id': 'timeseries-toggle-container'},
                                    ui.input_checkbox("historical_checkbox", "Historical", True),
                                    ui.input_checkbox("forecast_checkbox", "Forecast", True),
                                ),

                                ui.card({'id': 'timeseries-inner-container'},
                                    ui.output_plot('timeseries', width='100%', height='100%'),
                                ),
                            ),

                            ui.output_ui('show_time_slider'),
                            
                            ui.div({'id': 'download-csv-container', 'class': 'download-container'},
                                ui.download_link("download_csv_link", 'Download CSV')
                            ),
                            ui.div({'id': 'timeseries-table-container'},
                                ui.output_data_frame("timeseries_table"),
                            ),

                            ui.busy_indicators.options(),
                        ),

                        # forecast map tab
                        ui.nav_panel('Forecast map', 
                            ui.div({'id': 'forecast-map-container'},
                                ui.output_ui('forecast_map'),
                            ),
                        ),

                        id='tab_menu'
                    ),
                
                ),
            ),

            ui.panel_conditional(
                "input.about_button > input.close_about_button",
                ui.div({'id': 'about-inner-container'},
                    ui.div({'id': 'about-header'},
                        ui.input_action_button("close_about_button", "X"),
                    ),
                    ui.div({'id': 'about-body'},
                        ui.markdown(
                            """
                            ## Cooling degree days
                            This site displays  an **estimate** of historical cooling degree days (CDD) along with an experimental 6-month forecast. 
                            Note that the CDD metric is normally calculated with daily data and aggregated at the monthly or yearly level, whereas we
                            are attempting to estimate monthly CDD from monthly temperature data.

                            ## Data sources
                            The CDD layers were created using <a href="https://cds.climate.copernicus.eu/stac-browser/collections/reanalysis-era5-single-levels-monthly-means?.language=en" target="_blank">ERA5 monthly averaged data</a>.

                            National and state outlines were downloaded from <a href="https://www.naturalearthdata.com/" target="_blank">Natural Earth</a>. 

                            ## Woodwell Risk
                            You can find out more about the Woodwell Risk group and the work that we do on our <a href="https://www.woodwellclimate.org/research-area/risk/" target="_blank">website</a>. 
                            Whenever possible, we publish our <a href="https://woodwellrisk.github.io/" target="_blank">methodologies</a> and <a href="https://github.com/WoodwellRisk" target="_blank">code</a> on GitHub.
                            """
                        ),
                    ),
                ),
                {'id': 'about-container'},
            ), 

            ui.output_ui('show_update_message'),
        ),
    ),
)

def server(input: Inputs, output: Outputs, session: Session):
    
    countries_list = sorted(countries.name.values)
    country_options = reactive.value(countries_list)
    state_options = reactive.value([])

    country_name = reactive.value('')
    state_name = reactive.value('')
    filter_text = reactive.value('')
    bounds = reactive.value([])
    bbox = reactive.value([])

    # these values represent the data clipped to a specific area, used for the timeseries figures
    historical_cdd = reactive.value(None)
    forecast_cdd = reactive.value(None)

    # values for quickly storing and downloading figures and tables
    timeseries_to_save = reactive.value(None)
    table_to_save = reactive.value(None)
    add_download_links = reactive.value(True)

    slider_date = reactive.value(min_slider_date)


    @render.ui
    def show_update_message():
        if updating:
            return ui.TagList(
                ui.div({'id': 'update-message-container'},
                    ui.div({'id': 'update-message'}, 
                        'The website is currently being updated. Please check back later.'
                    ),
                ),
            )


    @reactive.effect
    @reactive.event(input.about_button)
    def action_button_click():
        ui.update_action_button("about_button", disabled=True)


    @render.ui
    def show_time_slider():
        if not updating:
            return ui.TagList(
            ui.panel_conditional('input.historical_checkbox == true',
                ui.div({'id': 'time-slider-container'}, 
                    ui.input_action_link('skip_months_button', 'Last 5 months', class_='skip-button'),
                    ui.input_action_link('skip_years_button', 'Last 5 years', class_='skip-button'),
                    ui.input_action_link('reset_skip_button', 'All data', class_='skip-button'),

                    ui.div({'id': 'time-slider-labels-container'},
                        ui.div({'class': 'time-slider-label'}, min_slider_date),
                        ui.output_text('time_slider_output'),
                        ui.div({'class': 'time-slider-label'}, max_slider_date),
                    ),
                    ui.input_slider('time_slider', '',
                        min=0,
                        max=len(dates) - 1,
                        value=skip_index,
                    ),
                ),
            {'id': 'show-slider-container'},
            ),
        )

    
    @reactive.effect
    @reactive.event(input.reset_skip_button)
    def reset_skip_button():
        ui.update_slider('time_slider', value=min_index)

    
    @reactive.effect
    @reactive.event(input.skip_years_button)
    def update_skip_years_button():
        ui.update_slider('time_slider', value=skip_index)

    
    @reactive.effect
    @reactive.event(input.skip_months_button)
    def update_skip_years_button():
        ui.update_slider('time_slider', value=max_index)


    @reactive.effect
    @reactive.event(input.time_slider)
    def update_slider_date():
        slider_date.set(dates[input.time_slider()])


    @render.text
    def time_slider_output():
        return slider_date()


    @reactive.effect
    @reactive.event(input.about_button)
    def action_button_click():
        ui.update_action_button("about_button", disabled=True)


    @reactive.effect
    @reactive.event(input.close_about_button)
    def action_button_close_click():
        ui.update_action_button("about_button", disabled=False)
    

    @reactive.effect
    @reactive.event(input.country_filter)
    def update_filter_text():
        filter_text.set(input.country_filter())


    @render.text
    def country_filter_text():
        return filter_text()


    @reactive.effect
    @reactive.event(filter_text)
    def update_country_list():
        query = filter_text()
        country_options.set(countries_list if query == '' else [value for value in countries_list if query.lower() in value.lower()])
    

    @reactive.effect
    @reactive.event(country_options)
    def update_country_select():
        new_options = country_options()
        ui.update_select('country_select', label=None, choices=new_options, selected=None)


    @reactive.effect
    @reactive.event(input.country_select)
    def update_country_name():
        new_country = input.country_select()
        country_name.set(new_country)
        state_name.set('')

    
    @reactive.effect
    @reactive.event(country_name)
    def update_state_list():
        cname = country_name()

        if(cname == ''): return

        df = states.query(" country == @cname ")
        states_list = sorted(df.name.values.tolist())
        # some countries have no administrative states / regions
        if(len(states_list) == 0 ):
            new_options = ['All']
        else: 
            if(cname == 'USA'):
                states_list = [state for state in states_list if state != 'CONUS']
                new_options = ['All', 'CONUS'] + states_list
            else:
                new_options = ['All'] + states_list
        
        state_options.set(new_options)

    @reactive.effect
    @reactive.event(state_options)
    def update_state_select():
        new_options = state_options()
        ui.update_select('state_select', label=None, choices=new_options, selected=None)


    @reactive.effect
    @reactive.event(input.state_select)
    def update_state_name():
        new_state = input.state_select()
        state_name.set(new_state)
    

    @reactive.effect
    @reactive.event(country_name, state_name)
    def update_bounds():
        cname = country_name()
        sname = state_name()

        # on app start or page reload, these variables will be empty
        if(cname == '' or sname == ''):
            return

        # https://stackoverflow.com/questions/1894269/how-to-convert-string-representation-of-list-to-a-list#1894296
        if(sname == 'All'):
            new_bounds = json.loads(countries.query(" name == @cname ").bbox.values[0])
        else:
            new_bounds = json.loads(states.query(" name == @sname and country == @cname ").bbox.values[0])
        bounds.set(new_bounds)

        xmin, ymin, xmax, ymax = new_bounds
        new_bbox = create_bbox_from_coords(xmin, xmax, ymin, ymax)
        bbox.set(new_bbox)


    @render.text
    def country_bbox_text():
        return bounds()


    @reactive.effect
    @reactive.event(input.process_data_button)
    def update_cdd_data():

        cname = country_name()
        sname = state_name()
        historical = h
        forecast = f

        # on app start or page reload, these variables will be empty
        if(cname == '' or sname == '' or historical is None or forecast is None):
            return

        xmin, ymin, xmax, ymax = bounds.get()
        bounding_box = bbox()
        country = countries.query(" name == @cname ")
        state = states.query(" name == @sname and country == @cname ")

        # we have already filtered countries where we don't have data, so clipping by country extent 
        # should never produce a rioxarray.exceptions.NoDataInBounds error at this step
        if(sname == 'All'):
            historical = historical.rio.clip(country.geometry, all_touched=True, drop=True)
            forecast = forecast.rio.clip(country.geometry, all_touched=True, drop=True)
        else:
            historical = historical.rio.clip(state.geometry, all_touched=True, drop=True)
            forecast = forecast.rio.clip(state.geometry, all_touched=True, drop=True)

        historical_cdd.set(historical)
        forecast_cdd.set(forecast)


    @reactive.effect
    @reactive.event(historical_cdd, forecast_cdd, input.historical_checkbox, input.forecast_checkbox)
    def update_dataframe():
        cname = country_name()
        sname = state_name()

        show_historical = input.historical_checkbox()
        show_forecast = input.forecast_checkbox()

        historical = historical_cdd()
        forecast = forecast_cdd()

        # if the xarray data is empty (on initial load) or if the toggles controlling which datasets to show are both false, then return empty dataframe
        if((forecast is None and historical is None) or (show_forecast == False and show_historical == False)):
            df = pd.DataFrame({
                'country': [], 'state': [], 'type': [], 'time': [], 'degree days': [],
            })
        else:
            # include just historical
            if(show_historical == True and show_forecast == False):
                df = historical.mean(dim=['x', 'y']).drop_vars('spatial_ref').to_pandas().reset_index()
                df['cdd'] = df['cdd'].astype(float).round(4)
                # df['agree'] = np.nan
                df['time'] = df['time'].dt.date
                df.columns = ['time', 'degree days']
                df['country'] = cname
                df['state'] = sname
                df['type'] = 'historical'
                
                df = df[['country', 'state', 'type', 'time', 'degree days']].sort_values('time', ascending=False).reset_index(drop=True)

            
            # include just forecast
            elif(show_historical == False and show_forecast == True):
                df = forecast.mean(dim=['x', 'y']).drop_vars('spatial_ref').to_pandas().reset_index()
                # this is the 50% line in the forecast data
                df['cdd'] = df['cdd'].astype(float).round(4)
                df['time'] = df['time'].dt.date
                df.columns = ['time', 'degree days']
                df['country'] = cname
                df['state'] = sname
                df['type'] = 'forecast'
                
                df = df[['country', 'state', 'type', 'time', 'degree days',]].sort_values('time', ascending=False).reset_index(drop=True)


            # else both are active, include both
            else:
                df_historical = historical.mean(dim=['x', 'y']).drop_vars('spatial_ref').to_pandas().reset_index()
                df_historical['cdd'] = df_historical['cdd'].astype(float).round(4)
                df_historical['time'] = df_historical['time'].dt.date
                df_historical.columns = ['time', 'degree days']
                df_historical['country'] = cname
                df_historical['state'] = sname
                df_historical['type'] = 'historical'
                df_historical = df_historical[['country', 'state', 'type', 'time', 'degree days']]

                df_forecast = forecast.mean(dim=['x', 'y']).drop_vars('spatial_ref').to_pandas().reset_index()
                df_forecast['cdd'] = df_forecast['cdd'].astype(float).round(4)
                df_forecast['time'] = df_forecast['time'].dt.date
                df_forecast.columns = ['time', 'degree days']
                df_forecast['country'] = cname
                df_forecast['state'] = sname
                df_forecast['type'] = 'forecast'
                df_forecast = df_forecast[['country', 'state', 'type', 'time', 'degree days']]

                df = pd.concat([df_historical, df_forecast]).sort_values('time', ascending=False).reset_index(drop=True)

        table_to_save.set(df)
        return df


    @reactive.effect
    @reactive.event(table_to_save)
    def set_download_button_states():
        """
        If there is nothing to download, then we want to disable the user's ability to download empty figures and tables.
        """
        df = table_to_save()

        if df.empty:
            ui.remove_ui(selector="#download_timeseries_link")
            ui.remove_ui(selector="#download_csv_link")
            add_download_links.set(True)
        else:
            if(add_download_links()):
                ui.insert_ui(
                    ui.download_link("download_timeseries_link", 'Download timeseries'),
                    selector="#download-timeseries-container",
                    where="beforeEnd",
                ),
                ui.insert_ui(
                    ui.download_link("download_csv_link", 'Download CSV'),
                    selector="#download-csv-container",
                    where="beforeEnd",
                ),
                add_download_links.set(False)



    @render.ui
    # @render_widget
    def plotly_timeseries():  
        # https://plotly.com/python/range-slider/
        config = {
            'staticPlot': False, 
            'displaylogo': False, 
            # 'displayModeBar': False, 
            'scrollZoom': False,
            # 'modeBarButtonsToRemove': ['zoom', 'pan', 'select', 'lasso2d', 'toImage']
            # 'modeBarButtonsToRemove': ['pan', 'select', 'lasso2d', 'toImage']
              'toImageButtonOptions': {
                'format': 'png', # one of png, svg, jpeg, webp
                'filename': 'custom_image',
                'height': 440,
                'width': 900,
                'scale':6 # Multiply title/legend/axis/canvas sizes by this factor
            }

        }

        # Create figure
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=list(df.Date), y=list(df.High), line=dict(color="#1b1e23")))

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#ebebec')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#ebebec')

        # i could also look more at sliders in general: https://plotly.com/python/sliders/
        fig.update_layout(
            xaxis=dict(
                # the .rangeslider-container class has most of the event listeners
                rangeslider=dict(
                    visible=False,
                    # range=['1991-01-01', None]
                ),
                type = 'date',
                fixedrange = True,
            ),
            # https://community.plotly.com/t/how-i-can-disiable-zoom-and-other-functions/28318/5
            yaxis=dict(fixedrange = True),
            height=390,
            margin=dict(l=0, r=10, t=0, b=0),
            plot_bgcolor = '#f7f7f7',
            paper_bgcolor='#f7f7f7',
        )

        fig_html = fig.to_html(config=config)
        return ui.HTML(fig_html)
        # return fig


    @render.plot
    @reactive.event(table_to_save, slider_date)
    def timeseries(alt="A graph showing a timeseries of historical and forecasted cooling degree days"):
        historical = historical_cdd()
        forecast = forecast_cdd()

        if(historical is None or forecast is None):
            return

        show_historical = input.historical_checkbox()
        show_forecast = input.forecast_checkbox()
        filter_date = slider_date()

        df = table_to_save()
        # the single upper limit for all charts does not work well here because of the large range that not
        # every country experiences. instead, we will need to calculate and upper limit (to the nearest 100) for each country
        # https://stackoverflow.com/questions/8866046/python-round-up-integer-to-next-hundred
        if df.empty:
            upper_limit = 100
        else:
            upper_limit = int(np.ceil(df['degree days'].max() / 50.0)) * 50
        
        df = df.query(" @pd.to_datetime(@df['time'], format='%Y-%m-%d') >= @pd.Timestamp(@filter_date) ")
        # print(df.head(20))
        # print()

        timeseries_color = '#1b1e23'
        high_certainty_color = '#f4c1c1'
        medium_certainty_color = '#f69a9a'

        legend_options = {
            'xdata': [0],
            'ydata': [0],
        }

        historical_label = Line2D(color=timeseries_color, markerfacecolor=timeseries_color, label='Historical', linewidth=1, **legend_options, )
        forecast_label = Line2D(color=timeseries_color, markerfacecolor=timeseries_color, label='Forecast', linestyle='--', linewidth=1, **legend_options)
        legend_elements = []

        fig, ax = plt.subplots()

        # if both are true, we need to stitch together the historical and forecast timeseries
        if(show_historical == True and show_forecast == True):
            # this is the historical data plus the first entry for forecast data
            # df_historical = df.query(" type == 'historical' ")
            df_historical = df.iloc[6:, :]
            # print(df_historical[['country', 'time', 'percentile']])
            # print()

            # this is the 6-month forecast
            df_forecast = df.iloc[0:7, :]
            # print(df_forecast[['country', 'time', 'percentile']])
            # print()
            
            ax.plot(df_forecast['time'], df_forecast['degree days'], color=timeseries_color, linestyle='--')
            ax.plot(df_historical['time'], df_historical['degree days'], color=timeseries_color)

            legend_elements = [historical_label, forecast_label]

        if(show_historical == True and show_forecast == False):
            ax.plot(df['time'], df['degree days'], color=timeseries_color)

            if(len(df) == 5):
                ax.set_xticks([date for date in df.time.values])
            
            legend_elements = [historical_label]

        if(show_historical == False and show_forecast == True):
            ax.plot(df['time'], df['degree days'], color=timeseries_color, linestyle='--')

            legend_elements = [forecast_label]

        ax.set_xlabel('Time', fontproperties=ginto_medium)
        ax.set_ylabel('Cooling degree days', fontproperties=ginto_medium)

        # when there are 60 or more entries in the dataframe, 
        # the date labels along the x-axis get crowded and difficult to read
        if(len(df) <= 60):
            date_format = mdates.DateFormatter('%m-%y')
            ax.xaxis.set_major_formatter(date_format)

        # use custom fonts for x and y axes labels
        for label in ax.get_xticklabels():
            label.set_fontproperties(ginto)
    
        for label in ax.get_yticklabels():
            label.set_fontproperties(ginto)

        ax.margins(0, 0)
        ax.set_ylim(-0.05, upper_limit)

        if(not show_forecast and not show_historical):
            ax.set_xticks([0, 1, 2, 3, 4, 5])
            ax.set_xticklabels(['', '', '', '', '', ''])

        # you need both of these to change the colors
        fig.patch.set_facecolor('#f7f7f7')
        ax.set_facecolor('#f7f7f7')

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.25)

        if(len(legend_elements) > 0):
            fig.legend(handles=legend_elements, ncols=len(legend_elements), loc='lower center', bbox_to_anchor=(0 if len(legend_elements) == 3 else 0.025, 0, 1, 0.5), fontsize='small', facecolor='white', frameon=False)

        timeseries_to_save.set(fig)
        return fig


    @render.download(filename=lambda: f'cdd-timeseries-{country_name().lower()}-{"" if state_name() == "" else state_name().lower()}-{"historical" if input.historical_checkbox() else ""}-{"forecast" if input.forecast_checkbox() else ""}-{forecast_date}.png'.replace('\'', '').replace(' ', '-').replace('--', '-').replace('--', '-'))
    def download_timeseries_link():

        cname = country_name()
        sname = state_name()

        forecast = forecast_cdd()
        historical = historical_cdd()

        if(forecast is None and historical is None):
            return

        show_historical = input.historical_checkbox()
        show_forecast = input.forecast_checkbox()

        fig = timeseries_to_save()
        plt.figure(fig)
        ax = plt.gca()

        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        if(show_historical == True and show_forecast == False):
            historical_and_forecast_label = 'Historical'
        elif(show_historical == False and show_forecast == True):
            historical_and_forecast_label = 'Forecasted'
        elif(show_historical == True and show_forecast == True):
            historical_and_forecast_label = 'Historical and forecasted'

        title = f"{historical_and_forecast_label} cooling degree days for {sname + ', ' if sname != '' and sname != 'All' else ''}{cname}"

        ax.set_title(title, fontproperties=ginto_medium)

        fig.subplots_adjust(top=0.9)

        with io.BytesIO() as buffer:
            plt.savefig(buffer, format="png", dpi=300)
            yield buffer.getvalue()


    @render.ui
    @reactive.event(forecast_cdd)
    def forecast_map(alt="a map showing the borders of a country of interest"):

        cname = country_name()
        sname = state_name()
        # either country or state polygon should be used for centroid calculation, but we don't need both
        country = countries.query(" name == @cname ")
        state = states.query(" name == @sname and country == @cname ")
        forecast = forecast_cdd()

        if(cname == '' or forecast is None):
            return

        config = {
            # 'staticPlot': False, 
            'displaylogo': False, 
            # 'displayModeBar': False, 
            'scrollZoom': True,
            # 'modeBarButtonsToRemove': ['zoom', 'pan', 'select', 'lasso2d', 'toImage']
            'modeBarButtonsToRemove': ['pan', 'select', 'lasso2d', 'toImage']
        }

        centroid = country.centroid.values[0]
        bbox = json.loads(country.bbox.values[0])
        bounding_box = create_bbox_from_coords(*bbox).to_geo_dict()

        max_bounds = max(abs(bbox[0] - bbox[2]), abs(bbox[1] - bbox[3])) * 111
        zoom = 11 - np.log(max_bounds)

        df = forecast['cdd'].drop_vars('spatial_ref').to_dataframe().dropna().reset_index()
        df.columns = ['time', 'y', 'x', 'Degree days']

        forecast_dates = df.time.unique().tolist()
        formatted_dates = [date.strftime("%b-%Y") for date in forecast_dates]

        fig = px.scatter_map(
            data_frame = df, 
            lat = df.y, 
            lon = df.x, 
            color = df['Degree days'],
            # https://plotly.com/python/builtin-colorscales/
            color_continuous_scale = 'agsunset',
            range_color = [0, 1400],
            hover_data = {'time': False, 'x': False, 'y': False, 'Degree days': ':.3f'},
            map_style = 'carto-darkmatter-nolabels', # 'carto-darkmatter-nolabels',
            zoom=zoom,
            height=445,
            animation_frame = 'time'
        )

        fig["layout"].pop("updatemenus")

        steps = []
        for idx in range(len(formatted_dates)):
            step = dict(
                method='animate',
                label=formatted_dates[idx]
            )
            steps.append(step)

        fig.update_layout(
            sliders=[{
                'currentvalue': {'prefix': 'Time: '},
                'len': 0.8,
                'pad': {'b': 10, 't': 0},
                'steps': steps,
                # 'transition': {'easing': 'circle-in'},
                'bgcolor': '#f7f7f7',
                'bordercolor': '#1b1e23',
                'activebgcolor': '#1b1e23',
                'tickcolor': '#1b1e23',
                'font': {'color': '#1b1e23', 'family': 'Ginto normal'},
            }],
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='#f7f7f7',
        )

        fig.update_coloraxes(
            colorbar_title_side='right',
            colorbar_title_font=dict(color='#f7f7f7', family='Ginto normal'),
            colorbar_len=0.8,
            colorbar_thickness=20,
            colorbar_tickfont=dict(color='#f7f7f7', family='Ginto normal'),
        )

        fig.update_layout(
            coloraxis_colorbar_x=0.01,
            hoverlabel=dict(font_family='Ginto normal')
        )

        fig.add_traces(
            px.scatter_geo(geojson=bounding_box).data
        )

        # figurewidget = go.FigureWidget(fig)
        # return figurewidget
        # return fig

        # https://stackoverflow.com/questions/78834353/animated-plotly-graph-in-pyshiny-express
        """
        The below is not working in CSS: the text font-family will not update.
        So I could try to directly change the HTML string to change the text in myself.

        .maplibregl-ctrl-attrib-inner {
            font-family: 'Ginto normal'
        }
        """

        fig_html = fig.to_html(config=config, auto_play=False)

        # to save individual images later: https://github.com/plotly/plotly.py/issues/664
        return ui.HTML(fig_html)
         

    @render.data_frame
    @reactive.event(table_to_save)
    def timeseries_table():
        df = table_to_save()

        return render.DataTable( df, width='100%', height='375px', editable=False, )
    

    @render.download(filename=lambda: f'cdd-table-{country_name().lower()}-{"" if state_name() == "" else state_name().lower()}-{"historical" if input.historical_checkbox() else ""}-{"forecast" if input.forecast_checkbox() else ""}-{forecast_date}.csv'.replace('\'', '').replace(' ', '-').replace('--', '-').replace('--', '-'))
    def download_csv_link():
        df = table_to_save()

        with io.BytesIO() as buffer:
            df.to_csv(buffer)
            yield buffer.getvalue()


app = App(app_ui, server, static_assets=static_dir)