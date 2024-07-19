import numpy as np
import os
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import lightgbm as lgb
import requests
from io import StringIO

# PRETRAINED MODELS FUNCTIONS
def heading0_model():
    print(" ---- Part 1. Predicting Heading0 (measured in Degrees). ---- ", "\n")
    directory = "./Models"
    model_heading_dir = os.path.join(directory, "heading0_model.pkl")
    heading0_model = joblib.load(model_heading_dir)
    return heading0_model

def hitbool_model():
    print(" ---- Part 2. Predicting Hitbool. ---- ", "\n")
    directory = "./Models"
    model_hitbool_dir = os.path.join(directory, 'light_gbm_hitbool_model.txt')
    lgb_hitbool_model = lgb.Booster(model_file=model_hitbool_dir)
    return lgb_hitbool_model

def drift_model():
    print(" ---- Part 3. Predicting Final Position. ---- ", "\n")
    directory = "./Models"
    model_drift_dir = os.path.join(directory, "drift_prediction_multi_rf.pkl")
    drift_model = joblib.load(model_drift_dir)
    return drift_model

# REST OF FUNCTIONS

def load_latest_file(folder_path, start_word):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    
    # Filter out non-files (e.g., subdirectories)
    files = [file for file in files if file.startswith(start_word)]
    
    # Sort files by modification time (most recent first)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)
    
    # Take the most recent file
    latest_file = files[0]
    
    # Construct the full path to the latest file
    latest_file_path = os.path.join(folder_path, latest_file)

    # Read the file into a pandas DataFrame
    df_original = pd.read_csv(latest_file_path, sep=",", decimal=".")  # Assuming it's a CSV file, adjust accordingly for other formats

    return df_original

def encode_angles(df_0: pd.DataFrame, angular_cols: list) -> pd.DataFrame:
    for col in angular_cols:
        print(f" Encoding angular variable ---> {col}")
        df_0["cos_" + col] = np.cos(np.radians(df_0[col]))
        df_0["sin_" + col] = np.sin(np.radians(df_0[col]))
        print(f" Encoded angular variable ---> {col}")
    return df_0

def process_input_file(df_1: pd.DataFrame) -> pd.DataFrame:
    angular_cols = [
    "WindDir", "WaveDir", "CurrDir",
    ]
    df_final = encode_angles(df_0=df_1, angular_cols=angular_cols)
    print(f"Attempting to predict {df_final.shape[0]} events using {df_final.columns}")
    return df_final

def process_angle_results(y_pred, time_series):
    angle_radians = np.arctan2(y_pred[:, 0], y_pred[:, 1])
    angle_degrees = np.degrees(angle_radians)
    angle_degrees_predicted = (angle_degrees + 360) % 360
    for index, angle in enumerate(angle_degrees_predicted):
        print(f" For timestamp {str(time_series.iloc[index])} heading predicted is {np.round(angle, 2)}º")
        


def plot_xy(df_drift, df_predict):
    """
    This function plots the x-y initial points and their final prediction.
    """
    # Convert 'TIME' to a numerical format representing day and hour
    df_predict['TIME'] = pd.to_datetime(df_predict['TIME'])
    df_drift['day'] = df_predict['TIME'].dt.day
    df_drift['hour'] = df_predict['TIME'].dt.hour
    
    fig, ax = plt.subplots(figsize=(10, 8))
    for index, row in df_drift.iterrows():
        if -row['yf'] < 500 and row['xf'] < 500:
            ax.scatter(-row['y0'], row['x0'],  color='blue', label='Initial Point' if index == 0 else None)
            ax.scatter(-row['yf'], row['xf'],  color='red', label='Closest Point' if index == 0 else None)
            ax.plot([-row['y0'], -row['yf']], [row['x0'], row['xf']],  color='green', linestyle='dashed', label='Connector' if index == 0 else None)
            plt.annotate(f"{row['day']}\n\n{row['hour']}", xy=(-row['yf'], row['xf']), ha='center', va='center', color='black', fontsize=8) 
            
        else:
            ax.scatter(-row['y0'], row['x0'],  color='blue', label='Initial Point' if index == 0 else None)
            ax.scatter(-row['yf'], row['xf'],  color='red', label='Final Point' if index == 0 else None)
            ax.plot([-row['y0'], -row['yf']], [row['x0'], row['xf']],  color='green', linestyle='dashed', label='Trajectory' if index == 0 else None)
    
    ax.set_xlabel("Coordinate X")
    ax.set_ylabel("Coordinate Y")
    ax.legend()
    plt.plot(0, 0, marker='*', markersize=20, color='red')
    plt.ylim(-200, 1200)
    plt.xlim(-200, +1500)
    plt.annotate("Platform", xy=(70, 70), fontsize=12, color='red', ha='center', va='center')
    X_SLS = 1170
    Y_SLS = +765
    plt.annotate("SLS", xy=(X_SLS, Y_SLS), fontsize=12, color='black', ha='center', va='center', weight='bold')
    plt.plot(X_SLS, Y_SLS, marker="h", markersize=16, color='green')
    plt.title("Drift Representation")
    plt.show()

def process_hitbool_events(predicted_class, time_series, shap_values, explainer, hitbool_columns):
    for index, event in enumerate(predicted_class):
        if event == 0:
            print("\033[92m", f"Event {str(time_series[index])[:-4]} WILL NOT result in IMPACT", "\033[0m")
            shap.force_plot(explainer.expected_value[1], shap_values[1][index, :], matplotlib=True, feature_names=hitbool_columns)
        else:
            print("\033[91m", f"Event {str(time_series[index])[:-4]} may result in IMPACT", "\033[0m")
            shap.force_plot(explainer.expected_value[1], shap_values[1][index, :], matplotlib=True, feature_names=hitbool_columns)
        print(" ")    

# WE CREATE A FUNCTION TO FILTER HOUR DATAFRAME BASED ON DATE
def time_frame(df, column, day_range, hour_range):
    df[column] = pd.to_datetime(df[column])
    filtered_df = df[(df[column].dt.day >= day_range[0]) & (df[column].dt.hour > hour_range[0]) & (df[column].dt.hour <= hour_range[1]) & (df[column].dt.day < day_range[1])]
    return filtered_df


# TRANFORM THE WEATHER FORECAST DATA FORMAT
def transform_data(df_windwave):
    merged_df = df_windwave    
    # Filter columns
    column_position_to_check = 6
    desired_column_value = "FF10 [m/s]"
    columns_to_keep = [merged_df.columns[4], merged_df.columns[5], merged_df.columns[6], merged_df.columns[11], merged_df.columns[13], merged_df.columns[15], merged_df.columns[29], merged_df.columns[30]]


    # Check if the column name at a specific position equals a specific value
    if merged_df.columns[column_position_to_check] == desired_column_value:
        merged_df[str(merged_df.columns[column_position_to_check])] *= 1
    else:
        merged_df[str(merged_df.columns[column_position_to_check])] *= 0.5144

    # Use a certain columns_to_remove list
    columns_to_remove = [col for col in merged_df.columns if col not in columns_to_keep]

    # Add additional columns for the else case if needed
    df_clean = merged_df.drop(columns=columns_to_remove)

    # Rename columns
    column_mapping = {df_clean.columns[i]: ["TIME", "WindDir", "WindSpeed", "WaveHs", "WaveTp", "WaveDir", "CurrDir", "CurrSpeed"][i] for i in range(len(df_clean.columns))}
    df_clean = df_clean.rename(columns=column_mapping)

    # Reorder columns
    column_order = ["TIME", "WindSpeed", "WindDir", "WaveHs", "WaveTp", "WaveDir", "CurrSpeed", "CurrDir"]
    df_clean = df_clean[column_order]

    # Convert WindSpeed to meters per second
    # df_clean['WindSpeed'] *= 0.5144

    # Convert columns to appropriate data types
    column_types = {"WindSpeed": float, "WaveHs": float, "WaveTp": float}
    df_clean = df_clean.astype(column_types)

    # Specify the column for which you want to find the highest and lowest values
    value_column = 'TIME'  # Replace with the desired column name

    # Find the highest and lowest values in the specified column
    max_value = df_clean[value_column].max()
    min_value = df_clean[value_column].min()

    # Extract day and month from the timestamp column
    df_clean['TIME'] = pd.to_datetime(df_clean['TIME'])
    max_date = df_clean.loc[df_clean[value_column].idxmax(), 'TIME']
    min_date = df_clean.loc[df_clean[value_column].idxmin(), 'TIME']
    max_day_month = max_date.strftime('%d-%m')
    min_day_month = min_date.strftime('%d-%m')

    # Construct the custom file name
    custom_file_name = f'yme_weather_{value_column}_{min_day_month}_{max_day_month}.csv'
    
    # Optional: Save the resulting DataFrame to a CSV file
    save_path = f'./output/YME_weather/{custom_file_name}'  # Update with the desired path
    df_clean.to_csv(save_path, index=False)
    print(f"CSV file saved at: {save_path}")

    return df_clean


# FUNCTION TO TRANSFORM THE DIRECTIONAL VARIABLES TO THE RIGHT ANGLE VALUE
def transform_dir_variables(df: pd.DataFrame, df_bouy: pd.DataFrame) -> pd.DataFrame:

    df_1 = df_bouy.copy()
    df_2 = df.copy()

    if len(df_1) != 0:
        
        df_1_row = df_1.loc[df_1['TIME'] == df_1['TIME'].max()]
        df_2_row = df_2.iloc[[0]]
        print("The timestamp of the Bouy data ", df_1_row['TIME'])
        print("The timestamp of the weather data ", df_2_row['TIME'])

        df_adjust_values = pd.DataFrame(data=0, columns=df_2.columns, index=df_2.index).iloc[[0]]
        df_adjust_values["WindSpeed"] = df_2_row["WindSpeed"] - df_1_row["WindSpeed"]
        df_adjust_values["WindDir"] = df_2_row["WindDir"] - df_1_row["WindDir"]
        df_adjust_values["CurrSpeed"] = df_2_row["CurrSpeed"] - df_1_row["CurrSpeed"]
        df_adjust_values["CurrDir"] = df_2_row["CurrDir"] - df_1_row["CurrDir"]
        df_adjust_values["WaveHs"] = df_2_row["WaveHs"] - df_1_row["WaveHs"]
        
        
        # df_2["WindSpeed"] = pd.to_numeric(df_2["WindSpeed"], errors='coerce')  # 'coerce' will replace non-numeric values with NaN
        # df_2["WindDir"] = pd.to_numeric(df_2["WindDir"], errors='coerce')  # 'coerce' will replace non-numeric values with NaN
        # df_2["CurrSpeed"] = pd.to_numeric(df_2["CurrSpeed"], errors='coerce')  # 'coerce' will replace non-numeric values with NaN
        # df_2["CurrDir"] = pd.to_numeric(df_2["CurrDir"], errors='coerce')  # 'coerce' will replace non-numeric values with NaN
        # df_2["WaveHs"] = pd.to_numeric(df_2["WaveHs"], errors='coerce')  # 'coerce' will replace non-numeric values with NaN
        # df_2["WaveTp"] = pd.to_numeric(df_2["WaveTp"], errors='coerce')  # 'coerce' will replace non-numeric values with NaN
        # # Transform WindDir and WaveDir
        # df_2["WindDir"] = ((df_2["WindDir"] + 180) % 360) + 1.3  #THE DIRECTION IS FROM THE PLATFORM, TRANSFORM TOWARDS HERE
        # df_2["WaveDir"] = ((df_2["WaveDir"] + 180) % 360) + 1.3 #THE DIRECTION IS FROM THE PLATFORM, TRANSFORM TOWARDS HERE
        # df_2["CurrDir"] = (df_2["CurrDir"]) + 1.3  #THE DATA ALREADY COMES TOWARDS THE PLATFORM
    
        # Now let's adjust the values with the Bouy Data
        df_2["WindSpeed"] = (df_2["WindSpeed"] * (100 - (df_adjust_values["WindSpeed"] / 100)))/100
        df_2["WindDir"] = (df_2["WindDir"] * (100 - (df_adjust_values["WindDir"] / 100)))/100
        df_2["CurrSpeed"] = (df_2["CurrSpeed"] * (100 - (df_adjust_values["CurrSpeed"] / 100)))/100
        df_2["CurrDir"] = (df_2["CurrDir"] * (100 - (df_adjust_values["CurrDir"] / 100)))/100
        df_2["WaveHs"] = (df_2["WaveHs"]* (100 - (df_adjust_values["WaveHs"] / 100)))/100
    else:
        df_2["WindSpeed"] = pd.to_numeric(df_2["WindSpeed"], errors='coerce')  # 'coerce' will replace non-numeric values with NaN
        df_2["WindDir"] = pd.to_numeric(df_2["WindDir"], errors='coerce')  # 'coerce' will replace non-numeric values with NaN
        df_2["CurrSpeed"] = pd.to_numeric(df_2["CurrSpeed"], errors='coerce')  # 'coerce' will replace non-numeric values with NaN
        df_2["CurrDir"] = pd.to_numeric(df_2["CurrDir"], errors='coerce')  # 'coerce' will replace non-numeric values with NaN
        df_2["WaveHs"] = pd.to_numeric(df_2["WaveHs"], errors='coerce')  # 'coerce' will replace non-numeric values with NaN
        df_2["WaveTp"] = pd.to_numeric(df_2["WaveTp"], errors='coerce')  # 'coerce' will replace non-numeric values with NaN
        # Transform WindDir and WaveDir
        df_2["WindDir"] = ((df_2["WindDir"] + 180) % 360) + 1.3  #THE DIRECTION IS FROM THE PLATFORM, TRANSFORM TOWARDS HERE
        df_2["WaveDir"] = ((df_2["WaveDir"] + 180) % 360) + 1.3 #THE DIRECTION IS FROM THE PLATFORM, TRANSFORM TOWARDS HERE
        df_2["CurrDir"] = (df_2["CurrDir"]) + 1.3  #THE DATA ALREADY COMES TOWARDS THE PLATFORM
        

    # WE ARE ADDING 1.3º TO ALIGN WITH GM USING THE GRID NORTH (WHICH HAS AN OFFSET) AND WEATHER DATA IS RELATIVE TO TRUE NORTH
    # THEREFOR THE RESULTS WILL BE RELATIVE TO GRID NORTH

    return df_2



# FUNCTION TO PREDICT THE HEADING0 AND ADD IT TO THE DATAFRAMES
def heading0_report(df_list, heading0_model):
    df_pred_list = []
    for df in df_list:
        ######################################################################################################
        # First Model: Heading0
        ######################################################################################################
        heading0_cols = [
            'cos_WindDir','sin_WindDir', 'cos_WaveDir', 'sin_WaveDir', 'cos_CurrDir',  'sin_CurrDir', 
            'WindDir', 'WaveDir', 'CurrDir', 'WindSpeed', 'WaveHs', 'WaveTp', 'CurrSpeed',
        ]

        df_heading0 = df[heading0_cols]
            
        heading0 = heading0_model

        y_pred_heading0 = heading0.predict(df_heading0)
        
        process_angle_results(y_pred_heading0,time_series = df["TIME"])

        # Let's add the heading0 sin,cos columns
        df["sin_heading0"], df["cos_heading0"] = y_pred_heading0[:, 0], y_pred_heading0[:, 1]

        # Calculate the angle in radians
        angle_radians = np.arctan2(y_pred_heading0[:, 0], y_pred_heading0[:, 1])
        angle_degrees = np.degrees(angle_radians)
        angle_degrees_predicted = (angle_degrees + 360) % 360
        
        df["heading0"] = np.round(angle_degrees_predicted, 2)

        df_pred_list.append(df)

    return df_pred_list


# FUNCTION TO PLOT CIRCLE DIAGRAM
def sls_diagram_plot(df, AIS_df, title):
    ##############################################################################################################
    # LET'S SETUP THE PLOT AND CIRCLE CONDITIONS FIRST
    
    # Define the degrees
    degrees = np.arange(0, 360, 1)
    
    # Convert degrees to radians
    radians = np.deg2rad(degrees)
    
    # Create the plot
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(radians, np.ones_like(radians))
    
    # Set the direction of 0 degrees to be at the top
    ax.set_theta_zero_location('N')
    
    # Set the direction of rotation to be clockwise
    ax.set_theta_direction(-1)
    
    # Set the number of radial ticks
    ax.set_yticks([])
    
    # Set the labels at every 30 degrees
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 30)))
    
    # Set the labels
    ax.set_xticklabels([f'{deg}\u00b0' for deg in range(0, 360, 30)])
    ##############################################################################################################
    
    ##############################################################################################################
    # Yellow Restricted Zone – 116, 20 on the Kongsberg vessel nav system would appear as 114.3, 18.3 on your map.  
    #However, since those zones have no identifiable basis.  
    #I recommend that we use 104.58, 13.32 consistent with the as-built, to a 1000m circle.
    
    #Orange Restricted Zone – I recommend you use 79.89, 38.01, again consistent with the as-built tangent lines to a 500m circle
    #############################################################################################################
    
    ##############################################################################################################
    # NOW WE ADD THE RESTRICTING CONDITIONS LINES
    
    # Plot the radial lines for the yellow zone, range 200 to 296 degrees -> apparently 193.32 and 284.58
    restric_angles = [193.32, 284.58]
    for angle in restric_angles:
        radian = np.deg2rad(angle)
        ax.plot([radian, radian], [0, 1], color='orange', linestyle='-', linewidth=3)
    
    # Plot the radial lines for the range 20 to 116 degrees -> apparently 13.32 and 104.58
    heading_angles = [13.32, 104.58]
    for angle in heading_angles:
        radian = np.deg2rad(angle)
        ax.plot([radian, radian], [0, 1], color='grey', linestyle='-', linewidth=3)
    
    # Plot the radial lines for the red zone 79.89, 38.01 + 180
    restric_angles = [79.89 + 180, 38.01 + 180]
    for angle in restric_angles:
        radian = np.deg2rad(angle)
        ax.plot([radian, radian], [0, 1], color='red', linestyle='-', linewidth=3)
    ##############################################################################################################
    
    ##############################################################################################################
    # FINALLY WE GET THE DATA TO PLOT FROM THE DATAFRAMES
    ### LET'S DO THE HEADING0 FIRST

    
    # Let's add the values to plot from the dataframe
    angles = df["heading0"]
    values = df['TIME'].dt.hour
    
    # Plot arrows and tags based on values from the dataframe ## ADDING 180 TO SHOW ARROWS ON THE OTHER SIDE
    for angle, value in zip(angles, values):
        value = np.round(value, 2)
        angle = np.round(angle, 2) + 180
        radian = np.radians(angle)
        ax.annotate('', xy=(radian, 1), xytext=(radian, value * 0.1),
                    arrowprops=dict(color='blue', arrowstyle='->'))
        ax.text(radian, value * 0.1, str(value), ha='left', va='bottom', color='blue')

    
    ### NOW THE REAL AIS HEADING
    if  AIS_df.empty:
        print("The AIS dataframe is empty")
    else:
        angles_AIS = AIS_df["trueHeading"]
        values = AIS_df['msgtime'].dt.hour
        
        # Plot arrows and tags based on values from the dataframe ## ADDING 180 TO SHOW ARROWS ON THE OTHER SIDE
        for angle, value in zip(angles_AIS, values):
            value = np.round(value, 2)
            angle = np.round(angle, 2) + 180
            radian = np.radians(angle)
            ax.annotate('', xy=(radian, 1), xytext=(radian, value * 0.1),
                        arrowprops=dict(color='purple', arrowstyle='->'))
            ax.text(radian, value * 0.1, str(value), ha='left', va='bottom', color='purple')
    
    
    wind_values = df["WindDir"]
    
    # Plot arrows and tags FOR WIND DIRECTION
    # I AM GOING TO ADD 180 TO CHANGE THE POSITION OF THE ARROW, FOR LESS CONFUSION (BUT THE VALUE SHOULD BE THE OPPOSITE)
    for angle, value in zip(wind_values, values):
        value = np.round(value, 2)
        angle = np.round(angle, 2) + 180
        radian = np.deg2rad(angle)
        ax.annotate('', xy=(radian, 1), xytext=(radian, value * 0.1),
                    arrowprops=dict(color='red', arrowstyle='->'))
        ax.text(radian, value * 0.1, str(value), ha='left', va='bottom', color='red')
    
    wave_values = df["WaveDir"]
    
    # Plot arrows and tags FOR WAVE DIRECTION
    # I AM GOING TO ADD 180 TO CHANGE THE POSITION OF THE ARROW, FOR LESS CONFUSION (BUT THE VALUE SHOULD BE THE OPPOSITE)
    for angle, value in zip(wave_values, values):
        value = np.round(value, 2)
        angle = np.round(angle, 2) + 180
        radian = np.deg2rad(angle)
        ax.annotate('', xy=(radian, 1), xytext=(radian, value * 0.1),
                    arrowprops=dict(color='green', arrowstyle='->'))
        ax.text(radian, value * 0.1, str(value), ha='left', va='bottom', color='green')
    
    
    current_values = df["CurrDir"]
    
    # Plot arrows and tags FOR CURRENT
    # I AM GOING TO ADD 180 TO CHANGE THE POSITION OF THE ARROW, FOR LESS CONFUSION (BUT THE VALUE SHOULD BE THE OPPOSITE)
    for angle, value in zip(current_values, values):
        value = np.round(value, 2)
        angle = np.round(angle, 2) 
        radian = np.deg2rad(angle)
        ax.annotate('', xy=(radian, 1), xytext=(radian, value * 0.1),
                    arrowprops=dict(color='cyan', arrowstyle='<-'))
        ax.text(radian, value * 0.1, str(value), ha='left', va='bottom', color='cyan')
    
    ####################################################################################################333

    
    ##############################################################################################################
    # FINALLY THE LEGEND
    
    colors = ["purple", 'blue', 'green', 'red', "cyan"]
    labels = ["AIS Heading", "Pred Heading", 'waves', 'wind', "current"]
    
    # Add a legend with custom text and colored lines
    plt.legend(handles=[plt.Line2D([0], [0], color=color, linewidth=3, label=label) for color, label in zip(colors, labels)],
               loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    
    # Plotting the SLS
    plt.annotate("SLS", xy=(0, 0), fontsize=12, color='black', ha='center', va='center', weight='bold')
    plt.plot(0, 0, marker="h", markersize=16, color='green')

    # Set the title
    plt.title(title, loc='left', pad=20)
    ##############################################################################################################
    # Show the plot
    plt.show()