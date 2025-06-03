import pandas as pd


def get_heatwave(air_temp):
    # Create a copy of the DataFrame to avoid modification issues
    air_temp = air_temp.copy()

    threshold = 27
    
    # Step 1: Identify if each day exceeds the threshold
    air_temp["above_threshold"] = air_temp["air_temp"] > threshold
    
    # Step 2: Compute rolling sum over 3-day periods
    air_temp["rolling_sum"] = air_temp["above_threshold"].rolling(window=3, min_periods=1).sum()
    
    # Step 3: Initialize the Heatwave column
    air_temp["Heatwave"] = False
    
    # Step 4: Iterate over the dataset and apply back-labeling when a heatwave is detected
    for i in range(len(air_temp)):
        if air_temp.iloc[i]["rolling_sum"] >= 3:  # If a heatwave is detected
            j = i
            while j >= 0 and air_temp.iloc[j]["above_threshold"]: 
                air_temp.iloc[j, air_temp.columns.get_loc("Heatwave")] = True
                j -= 1  

    # Saving only heatwave period
    heatwave_period = air_temp.loc[air_temp["Heatwave"] == True]

    return heatwave_period
