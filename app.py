import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pickle
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
from datetime import datetime
date = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

# ML Libraries
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.arima import AutoARIMA

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score



###########################################
##### OTHER FUNCTIONS ######
###########################################

def dataframe_upload():
    df = None
    with st.spinner('Wait for few seconds your data is processing...'):
        fileName = st.file_uploader("Upload Data", type=["csv"],
                            help="Upload data here that needs to be processed")
    if fileName is not None:
        # read the csv
        df = pd.read_csv(fileName, sep=",")
        # Show the data that we just uploaded
        with st.expander("Show uploaded data", expanded=False):
            st.dataframe(df)
    return df


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

def download_csv(df, file_to_be_saved):
    csv = convert_df(df)
    st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name=file_to_be_saved,
            mime='text/csv',
        )




###########################################
##### MACHINE LEARNING FUNCTIONS  ######
###########################################

def decision(val):
    if val["CI"]>=9:
        return "Normal"
    elif val["CI"]>6 and val["CI"]<9:
        return "Warning"
    else:
        return "Clogging possibility"

def process_dataframe(df):
    df["CI"] = df["CI"].astype("float")
    df["Decision"] = df.apply(lambda x: decision(x[["CI"]]), axis=1)
    return df


def forecastAL(df):
    # Here we will process the dataframe we recieved and
    y = df['Al']
    [y_trainal, y_testal] = temporal_train_test_split(y, test_size=20)
    fh = ForecastingHorizon(y_testal.index, is_relative=False)
    regressor = KNeighborsRegressor(n_neighbors=4)
    forecaster = make_reduction(regressor, window_length=5, strategy="recursive")
    #forecaster.fit(y_trainal)
    #forecaster = AutoARIMA(sp=5)
    forecaster.fit(y_trainal)
    y_predal = forecaster.predict(fh)
    #plot_ys(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"]);
    y_predal = forecaster.predict(fh)
    # fig = plot_series(y_train, y_test, y_pred, labels=["y_train1", "y_test1", "y_pred1"])
    df['y_trainal'] = y_trainal
    df['y_testal'] =  y_testal
    df['y_predal'] =  y_predal
    return df

def forecastCI(df):
    # Here we will process the dataframe we recieved and
    y = df['CI']
    [y_trainci, y_testci] = temporal_train_test_split(y, test_size=20)
    fh = ForecastingHorizon(y_testci.index, is_relative=False)
    regressor = KNeighborsRegressor(n_neighbors=2)
    forecaster = make_reduction(regressor, window_length=20, strategy="recursive")
    #forecaster.fit(y_train)
    #forecaster = AutoARIMA(sp=2)
    forecaster.fit(y_trainci)
    y_predci = forecaster.predict(fh)
    #plot_ys(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"]);
    y_predci = forecaster.predict(fh)
    # fig = plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
    df['y_trainci'] = y_trainci
    df['y_testci'] =  y_testci
    df['y_predci'] =  y_predci
    return df

def forecastS(df):
    # Here we will process the dataframe we recieved and
    y = df['S']
    [y_trains, y_tests] = temporal_train_test_split(y, test_size=20)
    fh = ForecastingHorizon(y_tests.index, is_relative=False)
    regressor = KNeighborsRegressor(n_neighbors=4)
    forecaster = make_reduction(regressor, window_length=5, strategy="recursive")
    #forecaster.fit(y_trainal)
    #forecaster = AutoARIMA(sp=5)
    forecaster.fit(y_trains)
    y_preds = forecaster.predict(fh)
    #plot_ys(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"]);
    y_preds = forecaster.predict(fh)
    # fig = plot_series(y_train, y_test, y_pred, labels=["y_train1", "y_test1", "y_pred1"])
    df['y_trainS'] = y_trains
    df['y_testS'] =  y_tests
    df['y_predS'] =  y_preds
    return df

def forecastMn(df):
    # Here we will process the dataframe we recieved and
    y = df['Mn']
    [y_trainmn, y_testmn] = temporal_train_test_split(y, test_size=20)
    fh = ForecastingHorizon(y_testmn.index, is_relative=False)
    regressor = KNeighborsRegressor(n_neighbors=4)
    forecaster = make_reduction(regressor, window_length=5, strategy="recursive")
    #forecaster.fit(y_trainal)
    #forecaster = AutoARIMA(sp=5)
    forecaster.fit(y_trainmn)
    y_predmn = forecaster.predict(fh)
    #plot_ys(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"]);
    y_predmn = forecaster.predict(fh)
    # fig = plot_series(y_train, y_test, y_pred, labels=["y_train1", "y_test1", "y_pred1"])
    df['y_trainMn'] = y_trainmn
    df['y_testMn'] =  y_testmn
    df['y_predMn'] =  y_predmn
    return df

def forecastSi(df):
    # Here we will process the dataframe we recieved and
    y = df['Si']
    [y_trainsi, y_testsi] = temporal_train_test_split(y, test_size=20)
    fh = ForecastingHorizon(y_testsi.index, is_relative=False)
    #regressor = KNeighborsRegressor(n_neighbors=4)
    #forecaster = make_reduction(regressor, window_length=5, strategy="recursive")
    #forecaster.fit(y_trainal)
    forecaster = AutoARIMA(sp=5)
    forecaster.fit(y_trainsi)
    y_predsi = forecaster.predict(fh)
    #plot_ys(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"]);
    y_predsi = forecaster.predict(fh)
    # fig = plot_series(y_train, y_test, y_pred, labels=["y_train1", "y_test1", "y_pred1"])
    df['y_trainMn'] = y_trainsi
    df['y_testMn'] =  y_testsi
    df['y_predMn'] =  y_predsi
    return df



def decision_model(X_predict):
    filename  = "trained_model.pkl"
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model.predict(X_predict)[0]

def decisionfunction(df):
    df.set_index("Cast", drop=True, inplace=True)
    df.drop("Temperature", axis=1, inplace=True)
    df = df[["C","Mn","Si","P", "S", "Cr", "Ni", "Mo", "V", "Al", "Ca", "CI"]]
    df["Signal_decision"] = df.apply(lambda x: decision_model(x.to_numpy().reshape(1, -1)), axis=1)
    return df


def add_categorical_label(dataframe):
    dataframe.reset_index(drop=False, inplace=True)

    idx = dataframe[dataframe["Signal_decision"] == "Normal"].index
    dataframe.loc[idx, "categorical_label"] = 1

    idx = dataframe[dataframe["Signal_decision"] == "Warning"].index
    dataframe.loc[idx, "categorical_label"] = 2

    idx = dataframe[dataframe["Signal_decision"] == "Clog"].index
    dataframe.loc[idx, "categorical_label"] = 3

    dataframe["categorical_label"] = dataframe["categorical_label"].astype(int)

    return dataframe

###########################################
##### PLOTS ######
###########################################
def plot1(df):
    fig = px.scatter(df, x="Cast", y="CI", title="Check condition of Carbon in steelgrade with respect to Castability", color="C")
    # Add shape regions
    fig.add_hrect(y0="9", y1="12", fillcolor="darkgreen", opacity=0.1, layer="below", line_width=0)
    return fig

def plot2(df):
    fig = px.scatter(df, x="Cast", y="CI", title="Check condition of Aluminum in steelgrade with respect to Castability", color="Al")
    # Add shape regions
    fig.add_hrect(y0="9", y1="12", fillcolor="darkgreen", opacity=0.1, layer="below", line_width=0)
    return fig

def plot3(df):
    fig = px.scatter(df, x="Cast", y="CI", title="Check condition of Sulphur in steelgrade with respect to Castability", color="S")
    # Add shape regions
    fig.add_hrect(y0="9", y1="12", fillcolor="darkgreen", opacity=0.1, layer="below", line_width=0)
    return fig


def plot4(df):
    fig = px.scatter(df, x="Cast", y="CI", title="Check condition of Manganese in steelgrade with respect to Castability", color="Mn")
    # Add shape regions
    fig.add_hrect(y0="9", y1="12", fillcolor="darkgreen", opacity=0.1, layer="below", line_width=0)
    return fig

def forplotAL(df):
    fig = px.scatter(df, x="Cast", y=[df['y_trainal'], df['y_testal'], df['y_predal']],
                     labels=["y_train", "y_test", "y_pred"], title="Approximate forecasted values of Al")
    fig = fig.add_hrect(y0=(df['Al'].mean() + 3 * df['Al'].std()), y1=(df['Al'].mean() + 3 * df['Al'].std()),
                  fillcolor="darkgreen", opacity=0.1, layer="below", line_width=0)
    return fig

def forplotCI(df):
    fig = px.scatter(df, x="Cast", y=[df['y_trainci'], df['y_testci'], df['y_predci']],
                     labels=["y_train", "y_test", "y_pred"], title="Approximate forecasted values of Castability")
    fig = fig.add_hrect(y0=(df['CI'].mean() + 3 * df['CI'].std()), y1=(df['CI'].mean() + 3 * df['CI'].std()),
                  fillcolor="darkgreen", opacity=0.1, layer="below", line_width=0)
    return fig

def forplotS(df):
    fig = px.scatter(df, x="Cast", y=[df['y_trainS'], df['y_testS'], df['y_predS']],
                     labels=["y_train", "y_test", "y_pred"], title="Approximate forecasted values of S")
    fig = fig.add_hrect(y0=(df['S'].mean() + 3 * df['S'].std()), y1=(df['S'].mean() + 3 * df['S'].std()),
                  fillcolor="darkgreen", opacity=0.1, layer="below", line_width=0)
    # fig = plot_series(df['y_train'], df['y_test'], df['y_pred'], labels=["y_train", "y_test", "y_pred"])
    return fig

def forplotMn(df):
    fig = px.scatter(df, x="Cast", y=[df['y_trainMn'], df['y_testMn'], df['y_predMn']],
                     labels=["y_train", "y_test", "y_pred"], title="Approximate forecasted values of Mn")
    fig = fig.add_hrect(y0=(df['Mn'].mean() + 3 * df['Mn'].std()), y1=(df['Mn'].mean() + 3 * df['Mn'].std()),
                  fillcolor="darkgreen", opacity=0.1, layer="below", line_width=0)
    return fig

def ternary_plot(df, a, b, c, title):
    ## THIS IS A SCALE DATA FOR PLOTTING
    fig = px.scatter_ternary(df, a=a, b=b, c=c,
                                 title=title)
    return fig


def plot_guage(df_return, idx):
    fig = go.Figure(go.Indicator(
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    value = df_return.loc[idx, 'categorical_label'],
                    mode = "gauge",
                    title = {'text': "{}".format(df_return.loc[idx, 'Signal_decision']), 'font_size': 40},
                    gauge = {'axis': {'range': [None, 3], 'showticklabels': True,
                                        'tickvals': [0, 1, 2, 3],
                                        'ticktext':["", "Normal", "Warning", "Clog"]},
                            'bar': {'color': "black"},
                            'steps' : [
                                {'range': [0, 1], 'color': "green"},
                                {'range': [1, 2], 'color': "yellow"},
                                {'range': [2, 3], 'color': "red"}],
                            }))
    st.plotly_chart(fig, use_container_width=True)




####################################################################################
###### STREAMLIT MAIN PAGE VISULIZZATION ######
####################################################################################
def before_vaccum_degassing():
    st.markdown("""
    <h2 style='text-align: center; color: black;'>
    Examining the Ladle Before Vacuum Analysis!
    </h2>""", unsafe_allow_html=True)

    # Figures to show
    fig1 = None
    fig2 = None
    fig3 = None
    fig4 = None
    file_to_be_saved = None
    df_return = None

    # Upload DataFrame
    st.write('Please upload ths steel composition consisting secondary metallurgy data at start of ladle refining')
    df = dataframe_upload()
    if df is not None:
        col1, col2, col3 = st.columns(3, gap="small")
        with col1:
            if st.button("Steelgrade 1"):
                st.markdown("Make sure you enter the data consiting of carbon **< 0.58 wt. %**")
                # Process the file
                df_return = process_dataframe(df)
                file_to_be_saved = "Steelgrade_1_{}.csv".format(date)
                # Figures
                fig1 = plot1(df_return)
                fig2 = plot2(df_return)
                fig3 = plot3(df_return)
                fig4 = plot4(df_return)

        with col2:
            if st.button("Steelgrade 2"):
                st.markdown("Make sure you enter the data consiting of carbon **< 1.5 wt. %**")
                # Process the file
                df_return = process_dataframe(df)
                file_to_be_saved = "Steelgrade_2_{}.csv".format(date)
                # Figures
                fig1 = plot1(df_return)
                fig2 = plot2(df_return)
                fig3 = plot3(df_return)
                fig4 = plot4(df_return)

        with col3:
            if st.button("Forecast"):
                # Figures
                forecastCI(df)[""]
                fig1 = forplotCI(forecastCI(df))
                fig2 = forplotAL(forecastAL(df))
                fig3 = forplotMn(forecastMn(df))
                fig4 = forplotS(forecastS(df))

        # Plots
        if fig1 is not None:
            st.plotly_chart(fig1, use_container_width=True)
        if fig2 is not None:
            st.plotly_chart(fig2, use_container_width=True)
        if fig3 is not None:
            st.plotly_chart(fig3, use_container_width=True)
        if fig4 is not None:
            st.plotly_chart(fig4, use_container_width=True)


        # DOWNLOAD CSV SHEET BUTTON
        if file_to_be_saved and df_return is not None:
            download_csv(df_return, file_to_be_saved)


def after_vaccum_degassing():
    st.markdown("""
    <h2 style='text-align: center; color: black;'>
    Examining the Ladle After Vacuum Analysis!
    </h2>""", unsafe_allow_html=True)

    # Figures to show
    fig1 = None
    fig2 = None
    fig3 = None
    fig4 = None
    file_to_be_saved = None
    df_return = None

    # Upload DataFrame
    st.write('Please upload ths steel composition consisting secondary metallurgy data at start of ladle refining')
    df = dataframe_upload()
    if df is not None:
        col1, col2, col3 = st.columns(3, gap="small")
        with col1:
            if st.button("Steelgrade 1"):
                st.markdown("Make sure you enter the data consiting of carbon **< 0.58 wt. %**")
                # Process the file
                df_return = process_dataframe(df)
                file_to_be_saved = "Steelgrade_1_{}.csv".format(date)
                # Figures
                fig1 = plot1(df_return)
                fig2 = plot2(df_return)
                fig3 = plot3(df_return)
                fig4 = plot4(df_return)

        with col2:
            if st.button("Steelgrade 2"):
                st.markdown("Make sure you enter the data consiting of carbon **< 1.5 wt. %**")
                # Process the file
                df_return = process_dataframe(df)
                file_to_be_saved = "Steelgrade_2_{}.csv".format(date)
                # Figures
                fig1 = plot1(df_return)
                fig2 = plot2(df_return)
                fig3 = plot3(df_return)
                fig4 = plot4(df_return)

        with col3:
            if st.button("Forecast"):
                # Figures
                fig1 = forplotCI(forecastCI(df))
                fig2 = forplotAL(forecastAL(df))
                fig3 = forplotMn(forecastMn(df))
                fig4 = forplotS(forecastS(df))

        # Plots
        if fig1 is not None:
            st.plotly_chart(fig1, use_container_width=True)
        if fig2 is not None:
            st.plotly_chart(fig2, use_container_width=True)
        if fig3 is not None:
            st.plotly_chart(fig3, use_container_width=True)
        if fig4 is not None:
            st.plotly_chart(fig4, use_container_width=True)

        # DOWNLOAD CSV SHEET BUTTON
        if file_to_be_saved and df_return is not None:
            download_csv(df_return, file_to_be_saved)


def tundish():
    st.markdown("""
    <h2 style='text-align: center; color: black;'>
    Examining the Tundish!
    </h2>""", unsafe_allow_html=True)

    # Figures to show
    fig1 = None
    fig2 = None
    fig3 = None
    fig4 = None
    fig5 = None
    file_to_be_saved = None
    df_return = None

    # Upload DataFrame
    st.write('Please upload ths steel composition consisting secondary metallurgy data at start of ladle refining')
    df = dataframe_upload()
    if df is not None:

        names_of_selection = ["Steelgrade 1", "Steelgrade 2", "Forecast", "Decision"]
        selected_option = st.radio('Select Option', names_of_selection, index=0)

        if  selected_option == "Steelgrade 1":
            st.markdown("Make sure you enter the data consiting of carbon **< 0.58 wt. %**")
            # Process the file
            df_return = process_dataframe(df)

            # Figures
            st.plotly_chart(plot1(df_return), use_container_width=True)
            st.plotly_chart(plot2(df_return), use_container_width=True)
            st.plotly_chart(plot3(df_return), use_container_width=True)
            st.plotly_chart(plot4(df_return), use_container_width=True)

            # DOWNLOAD CSV SHEET BUTTON
            file_to_be_saved = "Steelgrade_1_{}.csv".format(date)
            download_csv(df_return, file_to_be_saved)


        if  selected_option == "Steelgrade 2":
            st.markdown("Make sure you enter the data consiting of carbon **< 1.5 wt. %**")
            # Process the file
            df_return = process_dataframe(df)

            # Figures
            st.plotly_chart(plot1(df_return), use_container_width=True)
            st.plotly_chart(plot2(df_return), use_container_width=True)
            st.plotly_chart(plot3(df_return), use_container_width=True)
            st.plotly_chart(plot4(df_return), use_container_width=True)

            # DOWNLOAD CSV SHEET BUTTON
            file_to_be_saved = "Steelgrade_2_{}.csv".format(date)
            download_csv(df_return, file_to_be_saved)


        if  selected_option == "Forecast":
                # Figures
                forcast_ci = forecastCI(df)["y_predci"].dropna().reset_index(drop=True)
                forcast_al = forecastAL(df)["y_predal"].dropna().reset_index(drop=True)
                forcast_mn = forecastMn(df)["y_predMn"].dropna().reset_index(drop=True)
                forcast = pd.concat([forcast_ci, forcast_al, forcast_mn], axis=1)

                # Data for ternary plot
                scaler = MinMaxScaler()
                forcast = forcast.astype("float")
                forcast[["y_predci", "y_predal", "y_predMn"]] = scaler.fit_transform(forcast)

                # Figures
                st.plotly_chart(forplotCI(forecastCI(df)), use_container_width=True)
                st.plotly_chart(forplotAL(forecastAL(df)), use_container_width=True)
                st.plotly_chart(forplotMn(forecastMn(df)), use_container_width=True)
                st.plotly_chart(forplotS(forecastS(df)), use_container_width=True)
                st.plotly_chart(ternary_plot(forcast, "y_predci", "y_predal", "y_predMn", "Forcasted Al and Mn with respect to CI"),)


        if selected_option == "Decision":
                df_return = decisionfunction(df)
                df_for_plotting = df_return.copy()
                df_for_plotting = add_categorical_label(df_for_plotting)

                idx = st.slider("Select Value", min_value=int(0), max_value=int(len(df_for_plotting)), value=int(0))
                with st.expander("Show", expanded=False):
                    st.dataframe(df_for_plotting)
                plot_guage(df_for_plotting, idx)

                # Ternary Plot
                scaler = MinMaxScaler()
                data_for_plot = df[["Al", "Ca", "Si"]]
                data_for_plot = data_for_plot.astype("float")
                data_for_plot[["Al", "Ca", "Si"]] = scaler.fit_transform(data_for_plot)
                st.plotly_chart(ternary_plot(data_for_plot, "Al", "Ca", "Si", "Probable NMI regions with respect to steel composition"), use_container_width=True)

                # DOWNLOAD CSV SHEET BUTTON
                file_to_be_saved = 'Decisions_modelresults_{}.csv'.format(date)
                download_csv(df_return, file_to_be_saved)

def main_page_title():
    col1, col2 = st.columns(2, gap="small")
    with col2:
        st.image("SIDENOR-logo.png")
    with col1:
        st.image("Checkcast.PNG")


def testing_page_content():
    # RADIO BUTTONS TO SELECT STAGE
    page_names_to_funcs = {
        "Before vacuum degassing": before_vaccum_degassing,
        "After vacuum degassing": after_vaccum_degassing,
        "Tundish": tundish,
    }
    selected_option = st.radio('Process Stage', page_names_to_funcs, index=0)
    page_names_to_funcs[selected_option]()


def training_page_content():
    st.markdown("""
    <h1 style='text-align: center; color: black;'>
    Training of Decision Model
    </h1>""", unsafe_allow_html=True)

    # Upload DataFrame
    st.write('Please upload data required for training the model')
    df = dataframe_upload()
    if df is not None:
        if "Decision" in df.columns:
            if st.button("Start Training Model"):
                with st.spinner('Wait for it...'):
                    clf = RandomForestClassifier()
                    X_data = df[["C", "Mn", "Si", "P", "S", "Cr", "Ni", "Mo", "V", "Al", "Ca", "CI"]].to_numpy()
                    y_data = df["Decision"].to_numpy()
                    clf.fit(X_data, y_data)
                    predictions = clf.predict(X_data)
                accuracy = accuracy_score(predictions, y_data, normalize=True)
                # Saving the model
                filename = 'trained_model.pkl'
                pickle.dump(clf, open(filename, 'wb'))
                st.markdown("The model is trained with accuracy of **{}%**".format(accuracy*100))
                st.markdown("The model saved as {}".format(filename))
        else:
            st.error("The loaded csv does not have columns 'Decision' !!.")


def sidebar():
    # Page in sidebar
    page_names_to_funcs = {
        "Testing": testing_page_content,
        "Training": training_page_content,
    }
    with st.sidebar:
        selected_page = st.radio("Select Tab", page_names_to_funcs.keys(), index=0)
    page_names_to_funcs[selected_page]()

def main():
    # Print LOGO on the top of the page
    main_page_title()
    # Sidebar page selection
    sidebar()


if __name__ =="__main__":
    main()
