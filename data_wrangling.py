#!/usr/bin/env python
# coding: utf-8
# # Meteodata from KEMIT HTML
# NB! Cannot use names with õüäö in URL request :(, need to use jaam_kood

# Andmed on võimalik masinloetaval kujul alla laadida avaandmete portaalist (kliimaandmete aegread) 
# https://keskkonnaportaal.ee/et/avaandmed/kliimaandmestik

import urllib.request, json 
import pandas as pd
import numpy as np
import pickle
import datetime as dt

import scipy.ndimage as ndimage

### ---------------------------------------------------
def station_codes(stations_l): 
    ### get names of stations from website
    # print(stations_l)
    url_str = "https://keskkonnaandmed.envir.ee/f_kliima_jaam_vaatlus"
    with urllib.request.urlopen(url_str) as url:
        jaamad = json.load(url)
    
    reff = pd.json_normalize(jaamad)
    df_stations = pd.DataFrame(data=reff)
    
    df_stat_uniq  = df_stations.loc[
        (df_stations["jaam_kood"] + df_stations["jaam_nimi"]).apply(sorted).drop_duplicates(keep="first").index, :
    ].reset_index(drop=True)


    ## create list of stations kood-s

    station_codes = []

    for j in stations_l:
        c = df_stat_uniq.loc[df_stat_uniq['jaam_nimi'] == j, 'jaam_kood'].values[0]
        station_codes.append(c)
       
    #print(df_stat_uniq[['jaam_nimi', 'jaam_kood']])
    
    ## create string of kood-s for URL
    stations_str = ''

    for i in station_codes:
        stations_str = stations_str + i    
        if i != station_codes[-1]:
            stations_str = stations_str +  ','
    return (stations_str)

### ---------------------------------------------------

### choose weather attributes  https://keskkonnaandmed.envir.ee/f_kliima_element

def get_data(year, start_month, end_month, el_kood, station_l, inajar, tyyp, outdir, date_style = 0):
    #print(station_l)
    stations_str = station_codes(station_l)
    
    url_str = "https://keskkonnaandmed.envir.ee/f_kliima_paev?aasta=eq.{}\
&kuu=gte.{}&kuu=lte.{}&element_kood=eq.{}&jaam_kood=in.({})".format(year, start_month, end_month, el_kood, stations_str)


    print(url_str)

    with urllib.request.urlopen(url_str) as url:
        data = json.load(url)
    
    reff = pd.json_normalize(data)
    df = pd.DataFrame(data=reff)
    print('normalised JSON data.....', df.shape)

    df.sort_values(['jaam_nimi', 'aasta', 'kuu','paev'],  inplace=True)


    i = 0
    for j in station_l:
   
        df_j = df.loc[df['jaam_nimi']==j].copy()
    
        df_j[['aasta', 'kuu', 'paev']] = df_j[['aasta', 'kuu', 'paev']].astype(str)

        #### vaikimisi date_style YYYY-MM-DD
        df_j['Date'] = pd.to_datetime(df_j.apply(lambda x:'%s-%s-%s' % (x['aasta'],x['kuu'],x['paev']),axis=1))

        if date_style == 1:    #### MM-DD
            df_j['Date']=df_j['Date'].astype(str).str[5: ]
        if date_style == 2:    
            df_j['Date']=df_j['Date'].dt.dayofyear 
    
        df_j = df_j.set_index('Date')
        df_j.drop(['paev' , 'kuu', 'aasta'], axis='columns',  inplace=True)
        df_j = df_j.rename(columns={'vaartus': j})
        df_j = df_j[[j]]
    
        df_j.head()
    
        if i == 0:
            results = df_j.copy()
        else:
            results= pd.merge(results, df_j, left_index=True, right_index=True)
        
        i =+ 1
        print(j, i)

    print('transpose data ...', results.shape)
    print('included dates ...', results.index)

 

    if inajar:
        if tyyp == 'pickle':        
            filename = outdir + year + start_month + end_month + '_' + el_kood + '.pickle'
            with open(filename, 'wb') as handle: 
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(filename,  end='\n')
            print('... purgis ....')
        if tyyp == 'csv':
            filename = outdir + year + start_month + end_month + '_' + el_kood + '.csv'
            results.to_csv(filename)
            print(filename,  end='\n')
            print('... excelis ....')

 
    
    
    return(results)

