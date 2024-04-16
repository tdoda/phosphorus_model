import os
import json
import math
import numpy as np
import pandas as pd
import gsw
import seawater as sw
from shutil import copyfile
from envass import qualityassurance
from datetime import datetime, timedelta
import time
from scipy.ndimage import uniform_filter1d


def copyFiles(outfolder, infolder):
    filelist = []
    for path, subdirs, files in os.walk(infolder):
        for name in files:
            filelist.append(os.path.join(path, name))

    copied = []
    for file in filelist:
        if ".TOB" in file and not os.path.isfile(os.path.join(outfolder, os.path.basename(file))):
            path_arr = os.path.basename(file).split(".")
            new_file = "{}__{}.{}".format(os.path.basename(os.path.dirname(file)), path_arr[0], path_arr[1])
            copyfile(file, os.path.join(outfolder, new_file))
            copied.append(file)
    return copied


def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return True


def check_valid_profile(data, value):
    if float(np.nanmax(data)) > float(value):
        return True
    else:
        return False


def strip_metadata(metadata):
    return metadata.replace(" ", "").split(":")[1]


def fixed_grid_resample_guide(data, grid):
    resample = []
    for g in grid:
        for j in range(len(data)):
            if data[j] > g or j >= len(data) - 1:
                resample.append({"index": False})
                break
            elif data[j] <= g < data[j + 1]:
                itp = (g - data[j]) / (data[j + 1] - data[j])
                resample.append({"index": j, "interpolation": itp})
                break
    return resample


def resample(guide, data):
    out = []
    for i in range(len(guide) - 1):
        if not guide[i]["index"]:
            out.append(np.nan)
        else:
            value = ((data[guide[i]["index"] + 1] - data[guide[i]["index"]]) * guide[i]["interpolation"]) + data[guide[i]["index"]]
            out.append(value)
    out.append(np.nan)
    return out


def index_of_max(arr):
    return np.argmax(np.array(arr))


def position_in_array(arr, value):
    for i in range(len(arr)):
        if value < arr[i]:
            return i
    return len(arr)


def round_to_days(dt, n):
    day = math.floor(dt.day / n)*n
    if day < 10:
        return "0" + str(day)
    else:
        return str(day)


def advanced_quality_flags(df, json_path="quality_assurance.json"):
    """
        input :
            - df is a dataframe of level 1B where basic check have been performed
            - json path: path for the advanced quality check json file, produced by the jupyter notebook
        output:
            - dictionnary where the dataframe is stored with updated advanced quality checks
        """
    quality_assurance_dict = json.load(open(json_path))
    var_name = quality_assurance_dict.keys()
    advanced_df = df.copy()
    for var in var_name:
        if quality_assurance_dict[var]:
            if var in advanced_df.keys(): 
                qa = qualityassurance(np.array(df[var]), np.array(df["time"]), **quality_assurance_dict[var]["advanced"])
                advanced_df[var + "_qual"].values[np.array(qa, dtype=bool)] = 1
    return advanced_df


def json_converter(qa):
    for keys in qa.keys():
        try:
            if qa[keys]["simple"]["bounds"][0] == "-inf":
                qa[keys]["simple"]["bounds"][0] = -np.inf
            if qa[keys]["simple"]["bounds"][1] == "inf":
                qa[keys]["simple"]["bounds"][1] = np.inf
        except:pass
    try:
        if qa["time"]["simple"]["bounds"][1] == "now":
            qa["time"]["simple"]["bounds"][1] = datetime.now().timestamp()
        return qa
    except:
        return qa
    
    
def log(str, indent=0, start=False,printlog=True):
    if printlog:
        if start:
            out = "\n" + str + "\n"
            with open("log.txt", "w") as file:
                file.write(out + "\n")
        else:
            out = datetime.now().strftime("%H:%M:%S.%f") + (" " * 3 * (indent + 1)) + str
            with open("log.txt", "a") as file:
                file.write(out + "\n")
        print(out)


def error(str):
    out = datetime.now().strftime("%H:%M:%S.%f") + "   ERROR: " + str
    with open("log.txt", "a") as file:
        file.write(out + "\n")
    raise ValueError(str)


def find_closest_index(arr, value):
    return min(range(len(arr)), key=lambda i: abs(arr[i] - value))


def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return True


def isnt_number(n):
    try:
        float(n)
    except ValueError:
        return True
    else:
        return False

def first_centered_differences(x, y, fill=False): 
    if x.size != y.size:
        log("first-centered differences: vectors do not have the same size")
    dy = np.full(x.size, np.nan)
    iif = np.where((np.isfinite(x)) & (np.isfinite(y)))[0]
    if iif.size == 0:
        return dy
    x0 = x[iif]
    y0 = y[iif]
    dy0 = np.full(x0.size, np.nan)
    dy0[0] = (y0[1] - y0[0]) / (x0[1] - x0[0])
    dy0[-1] = (y0[-1] - y0[-2]) / (x0[-1] - x0[-2])
    dy0[1:-1] = (y0[2:] - y0[0:-2]) / (x0[2:] - x0[0:-2])

    dy[iif] = dy0

    if fill:
        dy[0:iif[0]] = dy[iif[0]]
        dy[iif[-1] + 1:] = dy[iif[-1]]
    return dy


def default_salinity_temperature(temperature):
    return 1.8626 - 0.052908 * temperature + 0.00093057 * temperature ** 2 - 6.78e-6 * temperature ** 3

def fcond20_temperature_Kivu(temperature):
    # Compute f(T)=cond_20/cond(T)
    # Based on measurements in Lake Kivu by N. Gruber and A. Wüest in 2002
    return (-6E-06*temperature**3+0.0008*temperature**2-0.0465*temperature+1.6636)

def salinity(Temp, Cond, y_cond, temperature_func= default_salinity_temperature):
    ft = temperature_func(Temp)
    cond20 = ft * Cond * 1000 # uS/cm
    salin = y_cond * cond20 # g/kg
    return salin

def salinity_Kivu(Temp, Cond,temperature_func=fcond20_temperature_Kivu):
    # Compute salinity from conductivity
    # Based on measurements in Lake Kivu by N. Gruber and A. Wüest in 2002
    ft = temperature_func(Temp)
    cond20 = ft * Cond * 1000 # uS/cm
    salin=3E-08*cond20**2 + 0.001*cond20 - 0.0351 # g/kg
    return salin, cond20/1000

def density(temperature, salinity,press=0,C_CH4=0,C_CO2=0,beta_CH4=-1.25E-3,beta_CO2=0.25E-3):
    # C_CH4 and C_CO2 must be provided in g/L
    # Density from Chen & Millero (1986):
    rho = 1e3 * (
                0.9998395 + 6.7914e-5 * temperature - 9.0894e-6 * temperature ** 2 + 1.0171e-7 * temperature ** 3 -
                1.2846e-9 * temperature ** 4 + 1.1592e-11 * temperature ** 5 - 5.0125e-14 * temperature ** 6 + (
                    8.181e-4 - 3.85e-6 * temperature + 4.96e-8 * temperature ** 2) * salinity)
    # Approach: use the previous estimate of rho to calculate the next one (another option would be to use the same reference density for all estimates)
    if isinstance(C_CH4,np.ndarray) or (not C_CH4==0):
        rho=rho*(1+beta_CH4*C_CH4)
        
    if isinstance(C_CO2,np.ndarray) or (not C_CO2==0):
        rho=rho*(1+beta_CO2*C_CO2) 
        
    if isinstance(press,np.ndarray) or (not press==0) and (len(press)==len(temperature)):
        K=19652.17+148.113*temperature-2.293*temperature**2 + 1.256*1e-2*temperature**3\
 -4.18*1e-5*temperature**4+(3.2726-2.147*1e-4*temperature+1.128*1e-4*temperature**2)*press/10+(53.238-0.313*temperature+5.728*1e-3*press/10)*salinity
        rho=rho/(1-0.1*press/K)
        
        
    return rho


def density_Kivu(temperature, salinity,press=0,C_CH4=0,C_CO2=0,beta_S=0.75E-3,beta_CH4=-1.25E-3,beta_CO2=0.284E-3):
    # C_CH4 and C_CO2 must be provided in g/L
    # beta coefficients from Schmid et al., 2004
    rho_T = 1e3 * (
                0.9998395 + 6.7914e-5 * temperature - 9.0894e-6 * temperature ** 2 + 1.0171e-7 * temperature ** 3 -
                1.2846e-9 * temperature ** 4 + 1.1592e-11 * temperature ** 5 - 5.0125e-14 * temperature ** 6)
    
    contrib_S=beta_S*salinity
    # Approach: use the previous estimate of rho to calculate the next one (another option would be to use the same reference density for all estimates)
    if isinstance(C_CH4,np.ndarray) or (not C_CH4==0):
        contrib_CH4=beta_CH4*C_CH4
    else:
        contrib_CH4=0
        
    if isinstance(C_CO2,np.ndarray) or (not C_CO2==0):
        contrib_CO2=beta_CO2*C_CO2
    else:
        contrib_CO2=0
        
    rho=rho_T*(1+contrib_S+contrib_CH4+contrib_CO2)
        
    if isinstance(press,np.ndarray) or (not press==0) and (len(press)==len(temperature)):
        K=19652.17+148.113*temperature-2.293*temperature**2 + 1.256*1e-2*temperature**3\
 -4.18*1e-5*temperature**4+(3.2726-2.147*1e-4*temperature+1.128*1e-4*temperature**2)*press/10+(53.238-0.313*temperature+5.728*1e-3*press/10)*salinity
        rho=rho/(1-0.1*press/K)
        
        
    return rho



def Gamma_adiabatic(T, S, p, lat=46.):
    alpha = sw.alpha(S, T, p)
    cp = sw.cp(S, T, p)
    Gamma = sw.g(lat) * alpha * (T - 273.15) / cp
    return Gamma

def mask_single_data(data, mask):
    try:
        idx = mask > 0
        data = data.astype(float)
        data[idx] = np.nan
        return data
    except:
        print("Masking failed")
        return data


def potential_temperature(T, S, p, z, lat=46.2):
    iif = np.where(np.isfinite(T) & np.isfinite(S) & np.isfinite(p) & np.isfinite(z))
    PT = np.full(T.size, np.nan)
    T = T[iif]
    p = p[iif]
    z = z[iif]
    S = S[iif]
    pt0 = np.copy(T)
    n = pt0.size
    pt1 = np.full(n, np.nan)
    iterate = True
    j = 0
    while iterate:
        intGamma = np.zeros(n)
        for i in range(1, n):
            Gamma0 = Gamma_adiabatic(pt0, S[i], p, lat)
            intGamma[i] = np.trapz(Gamma0[0:i + 1], x=z[0:i + 1])
        pt1 = T + intGamma
        j += 1
        if j > 100 or np.nanmax(np.abs(pt1 - pt0)) < 1e-3:
            iterate = False
        else:
            pt0 = np.copy(pt1)

    PT[iif] = pt1
    return PT


def potential_temperature_gsw(T, S, p):
    return gsw.pt_from_t(S, T, p, 0)


def potential_temperature_sw(T, S, p, p_ref):
    """
    Calculates potential temperature as per UNESCO 1983 report.
    Parameters
    ----------
    s(p) : array_like
        salinity [psu (PSS-78)]
    t(p) : array_like
        temperature [℃ (ITS-90)]
    p : array_like
        pressure [db].
    pr : array_like
        reference pressure [db], default = 0
    Returns
    -------
    pt : array_like
        potential temperature relative to PR [℃ (ITS-90)]
    """
    return sw.ptmp(s=S,t=T,p=p,pr=p_ref)


def oxygen_saturation(T, S, altitude=372., lat=46.2, units="mgl"):
    # calculates oxygen saturation in mg/l according to Garcia-Benson
    # to be coherent with Hannah
    if units != "mgl" and units != "mll":
        units = "mgl"
    mgL_mlL = 1.42905
    mmHg_mb = 0.750061683
    mmHg_inHg = 25.3970886
    standard_pressure_sea_level = 29.92126
    standard_temperature_sea_level = 15 + 273.15
    gravitational_acceleration = gr = sw.g(lat)
    air_molar_mass = 0.0289644
    universal_gas_constant = 8.31447
    baro = (1. / mmHg_mb) * mmHg_inHg * standard_pressure_sea_level * np.exp(
        (-gravitational_acceleration * air_molar_mass * altitude) / (
                    universal_gas_constant * standard_temperature_sea_level))
    u = 10 ** (8.10765 - 1750.286 / (235 + T))
    press_corr = (baro * mmHg_mb - u) / (760 - u)

    Ts = np.log((298.15 - T) / (273.15 + T))
    lnC = 2.00907 + 3.22014 * Ts + 4.0501 * Ts ** 2 + 4.94457 * Ts ** 3 + -0.256847 * Ts ** 4 + 3.88767 * Ts ** 5 - S * (
                0.00624523 + 0.00737614 * Ts + 0.010341 * Ts ** 2 + 0.00817083 * Ts ** 3) - 4.88682e-07 * S ** 2
    O2sat = np.exp(lnC)
    if units == "mll":
        O2sat = O2sat * press_corr
    elif units == "mgl":
        O2sat = O2sat * mgL_mlL * press_corr

    return O2sat


def parse_file(input_file_path, string):
    # Define the parameters used to read the files based on the data after the selected string
    valid = True
    start_date=''
    with open(input_file_path, encoding="utf8", errors='ignore') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        if 'start_time' in lines[i]:
            start_date_str=lines[i][lines[i].find("start_time")+13:lines[i].find("[Instrument")-1]
            start_date=datetime.strptime(start_date_str,'%b %d %Y %H:%M:%S')
        if string in lines[i]:
            break
            print("yes")
    if input_file_path[-4:]=='.TOB':
        date_format = "%m/%d/%Y %H:%M:%S"
        columns = lines[i + 2].replace(";", "").split() 
        columns.pop(0)
        columns = rename_duplicates(columns)
        units = lines[i + 3].replace(";", "").replace("[", "").replace("]", "").split()
        skip_rows = i + 5
        n = 0
        while len(lines[i + 5].split()) - 1 > len(columns):
            columns.append(n)
            n = n + 1
        if len(lines) <= skip_rows + 1 or len(columns) < 5:
            valid=False
    elif input_file_path[-4:]=='.cnv':
        skip_rows=i+1
        # Should match the variable names and units of CTD class to save the variables
        columns=['Minutes','Depth','Temp','pH','Fluo','Cond','Flag'] 
        units=['min','m','degC','_','mg/m^3','uS/cm','_']
        valid=True
        date_format='%b %d %Y %H:%M:%S'
    return skip_rows, columns, units, valid, date_format, start_date, 

        
def rename_duplicates(arr):
    out = []
    d = {}

    for i in arr:
        d.setdefault(i, -1)
        d[i] += 1

        if d[i] >= 1:
            out.append('%s%d' % (i, d[i]))
        else:
            out.append(i)
    return out



def check_variable(variable, unit, columns, units):
    if variable in columns:
        for i in range(len(columns)):
            if variable == columns[i]:
                break
        if units[i] in unit: 
            return True
        else:
            log("{} needs unit [{}] but has unit [{}]".format(variable, unit, units[i]))
            return False
    else:
        return False

    
def parse_time(df, variable, name, columns, units, ref_date,day_month=True): 
    """
    Function description
    Structure:  
        - First level of if-else-statements checks if  AM or PM exists. 
        - Second level of if-else-statements checks what the column names for the date and time are.
        - The third level of if-else-statements is only triggered, if AM or PM exists and localizes in which column AM/PM 
        is located. The statement then adjusts the column headers of the dataframe by giving the column with AM/ PM the header "0
    Inputs:
        - day_month: if True, day is before month (only applied for format xx/xx/xxxx without AM/PM)         
    Output:
        New dataframe column with parsed time in it.
        Dataframe with adjusted column headers.
    """  
    AM_PM=["AM", "AM?", "AM.?", "PM", "PM?", "PM.?"]
    res = [ele for ele in AM_PM if(ele in df.values[0])] # Check if AM or PM or similar is present in the first row of the dataframe
   
    if "IntD" in columns and "IntT" in columns:     
        df=df.rename(columns = {'IntD':'Date', 'IntT':'Time'})
        columns[columns.index('IntD')]='Date'
        columns[columns.index('IntT')]='Time'
    elif "IntDT" in columns and "IntDT1" in columns:
        df=df.rename(columns = {'IntDT':'Date', 'IntDT1':'Time'})
        columns[columns.index('IntDT')]='Date'
        columns[columns.index('IntDT1')]='Time'
    elif "IntT" in columns and "IntT1" in columns:
        df=df.rename(columns = {'IntT':'Date', 'IntT1':'Time'})
        columns[columns.index('IntT')]='Date'
        columns[columns.index('IntT1')]='Time'
    elif "IntD" in columns and "IntD1" in columns:
        df=df.rename(columns = {'IntD':'Date', 'IntD1':'Time'})
        columns[columns.index('IntD')]='Date'
        columns[columns.index('IntD1')]='Time'
    else:
        raise ValueError("Cannot process unrecognised file.")
        
    if ":" in df["Date"][df.index[0]]: #Flip date and time
        hourdata=df["Date"]
        df["Date"]=df["Time"]
        df["Time"]=hourdata

    if bool(res)==True:
        dateformat="%m/%d/%Y %H:%M:%S"
            
        if bool([ele for ele in AM_PM if(ele in list(df["Time"]))])==True:
            df=df.rename(columns = {'Date':'Time', 'Time':'Date'}) # reverse time and date
            ind_date=columns.index('Date')
            ind_time=columns.index('Time')
            columns[ind_date]='Time'
            columns[ind_time]='Date'
        
        elif bool([ele for ele in AM_PM if(ele in list(df["Date"]))])==True: 
            # Invert column names
            columns[columns.index('Date')]='AMPM'
            columns[columns.index(0)]='Date'
            # del columns[-1]
            # columns.insert(columns.index("Date"), 0) 
            df.columns = columns
            if "?" in str([ele for ele in AM_PM if(ele in list(df["Date"]))]):
                df=df.replace({0:{'\?':'','\.':''}},regex=True) # Remove ? and .
            try:
                datetime_arr = pd.to_datetime(df["Date"] + " " + df["Time"], format=dateformat, dayfirst=True)
                arr = list(datetime_arr.values.astype(float) / 10 ** 9)
                # if ref_date and abs(arr[0] - ref_date) > 30*24*60*60:
                #     arr = list(
                #         datetime_arr.values.astype(float) / 10 ** 9)
                # df["time"] = arr
                # return df
            except Exception:
                log("Datetime file parse failed")
                raise
        
        elif bool([ele for ele in AM_PM if(ele in list(df[0]))])==True:
            if "?" in str([ele for ele in AM_PM if(ele in list(df[0]))]):
                df=df.replace({0:{'\?':'','\.':''}},regex=True) # Remove ? and .
            
            if ("-" in df["Date"][df.index[0]]):
                try:
                    datetime_arr = pd.to_datetime(df["Date"]+" "+df["Time"]+" "+df[0],format="%d-%b-%y %I:%M:%S %p") 
                except Exception:
                    try:
                        datetime_arr = pd.to_datetime(df["Date"]+" "+df["Time"]+" "+df[0],format="%d-%b-%y %I:%M:%S") 
                    except Exception:
                        raise Exception("Datetime file parse failed")    
            else:
                try:
                    datetime_arr = pd.to_datetime(df["Date"]+" "+df["Time"]+" "+df[0],format="%m/%d/%Y %I:%M:%S %p") 
                except Exception:
                    try:
                        datetime_arr = pd.to_datetime(df["Date"]+" "+df["Time"]+" "+df[0],format="%m/%d/%Y %I:%M:%S") 
                    except Exception:
                        raise Exception("Datetime file parse failed")  
            arr = list(datetime_arr.values.astype(float) / 10 ** 9)
            # df["time"] = arr
            # return df 
            # try:
            #     datetime_arr = pd.to_datetime(df["Date"]+" "+df["Time"]+" "+df[0],format="%m/%d/%Y %I:%M:%S %p")
            #     arr = list(datetime_arr.values.astype(float) / 10 ** 9)
            #     # if ref_date and abs(arr[0] - ref_date) > 30*24*60*60:
            #     #     arr = list(
            #     #         datetime_arr.values.astype(float) / 10 ** 9)
            #     df["time"] = arr
            #     return df
            # except Exception:
            #     datetime_arr = pd.to_datetime(df["Date"]+" "+df["Time"]+" "+df[0],format="%d-%b-%y %I:%M:%S %p")
            #     arr = list(datetime_arr.values.astype(float) / 10 ** 9)
            #     # if ref_date and abs(arr[0] - ref_date) > 30 * 24 * 60 * 60:
            #     #     arr = list(
            #     #         datetime_arr.values.astype(float) / 10 ** 9)
            #     df["time"] = arr
            #     return df
        else:
            del columns[-1]
            columns.insert(columns.index("Time")+1, 0)
            df.columns = columns
            units.insert(columns.index(0), 0)
            if "?" in str([ele for ele in AM_PM if(ele in list(df[0]))]):
                df=df.replace({0:{'\?':'','\.':''}},regex=True) # Remove ? and .

            try:
                try:
                    datetime_arr = pd.to_datetime(df["Date"]+" "+df["Time"]+" "+df[0],format="%m/%d/%Y %I:%M:%S %p")
                except Exception:
                    try:
                        datetime_arr = pd.to_datetime(df["Date"]+" "+df["Time"]+" "+df[0],format="%d-%b-%y %I:%M:%S %p")
                    except Exception:
                        datetime_arr = pd.to_datetime(df["Date"]+" "+df["Time"]+" "+df[0],format="%Y-%m-%d %I:%M:%S %p")
                arr = list(datetime_arr.values.astype(float) / 10 ** 9)
                # if ref_date and abs(arr[0] - ref_date) > 30*24*60*60:
                #     arr = list(
                #         datetime_arr.values.astype(float) / 10 ** 9)
                # df["time"] = arr
                # return df
            except Exception:
                breakpoint()
                log("Datetime file parse failed")
                raise
    else:           
            if ("-" in df["Date"][df.index[0]]) and df["Date"][df.index[0]].find("-")==2:
                try:
                    dateformat='%d-%b-%y %H:%M:%S'
                    arr = pd.to_datetime(df["Date"] + " " + df["Time"], format=dateformat).values.astype(float) / 10 ** 9
                except Exception:
                    try:
                        dateformat='%d-%b-%y %H:%M:%S %p'
                        arr = pd.to_datetime(df["Date"] + " " + df["Time"], format=dateformat).values.astype(float) / 10 ** 9
                    except Exception:
                        breakpoint()
                        raise Exception("Datetime file parse failed")    
            elif ("-" in df["Date"][df.index[0]]) and df["Date"][df.index[0]].find("-")==4:
                try:
                    dateformat='%Y-%m-%d %H:%M:%S'
                    arr = pd.to_datetime(df["Date"] + " " + df["Time"], format=dateformat).values.astype(float) / 10 ** 9
                except Exception:
                    try:
                        dateformat='%Y-%m-%d %H:%M:%S %p'
                        arr = pd.to_datetime(df["Date"] + " " + df["Time"], format=dateformat).values.astype(float) / 10 ** 9
                    except Exception:
                        breakpoint()
                        raise Exception("Datetime file parse failed")   
            else:
                try:
                    dateformat='%m/%d/%Y %H:%M:%S'
                    arr = pd.to_datetime(df["Date"] + " " + df["Time"], format=dateformat).values.astype(float) / 10 ** 9
                except Exception:
                    try:
                        dateformat='%m/%d/%Y %H:%M:%S %p'
                        arr = pd.to_datetime(df["Date"] + " " + df["Time"], format=dateformat).values.astype(float) / 10 ** 9
                    except Exception:
                        try: 
                            dateformat='%d/%m/%Y %H:%M:%S'
                            arr = pd.to_datetime(df["Date"] + " " + df["Time"], format=dateformat).values.astype(float) / 10 ** 9
                        except Exception:
                            try:
                                dateformat='%d/%m/%Y %H:%M:%S %p'
                                arr = pd.to_datetime(df["Date"] + " " + df["Time"], format=dateformat).values.astype(float) / 10 ** 9
                            except Exception:
                                breakpoint()
                                raise Exception("Datetime file parse failed")
                
                
                #arr = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=day_month,dayfirst=day_month).values.astype(float) / 10 ** 9
    if ref_date and abs(arr[0] - ref_date) > 30*24*60*60: # More than a month of difference between reference date --> invert day and month 
        ind_day=dateformat.find('%d')
        if '%b' in dateformat:
            month_format='%b'    
        else:
            month_format='%m'
        ind_month=dateformat.find(month_format) 
        dateformat=dateformat[:ind_day]+month_format+dateformat[ind_day+2:]
        dateformat=dateformat[:ind_month]+'%d'+dateformat[ind_month+2:]
        arr = pd.to_datetime(df["Date"] + " " + df["Time"], format=dateformat).values.astype(float) / 10 ** 9
    
    df["time"] = arr
    return df 
        
         
    #--------------------------------------------------
    # Previous code:      
                
        
    #     if "IntD" in columns and "IntT" in columns:            
    #         if bool([ele for ele in AM_PM if(ele in list(df["IntD"]))])==True:
    #             input("Press Enter to continue...")
    #             breakpoint()
    #             del columns[-1]
    #             columns.insert(columns.index("IntD"), 0) 
    #             df.columns = columns
    #             try:
    #                 #datetime_arr = pd.to_datetime(df["IntD"]+" "+df["IntT"]+" "+df[0],format="%m/%d/%Y %I:%M:%S %p")
    #                 datetime_arr = pd.to_datetime(df["IntD"] + " " + df["IntT"], format=dateformat, dayfirst=True)
    #                 try:
    #                     datetime_arr[df[df["IntD"] == "PM"].index] = datetime_arr[df[df["IntD"] == "PM"].index] + timedelta(hours=12)
    #                 except: pass
    #                 idx = np.argmin(np.diff(datetime_arr))
    #                 if np.diff(datetime_arr)[idx].astype("float")<0:
    #                     datetime_arr[idx+1:] = np.copy(datetime_arr[idx+1:]+timedelta(hours=12))
    #                 arr = list(datetime_arr.values.astype(float) / 10 ** 9)
    #                 if ref_date and abs(arr[0] - ref_date) > 30*24*60*60:
    #                     arr = list(
    #                         datetime_arr.values.astype(float) / 10 ** 9)
    #                 df["time"] = arr
    #                 return df
    #             except:
    #                 log("Datetime file parse failed")
    #                 raise
    #         if bool([ele for ele in AM_PM if(ele in list(df[0]))])==True:
    #             if "?" in str([ele for ele in AM_PM if(ele in list(df[0]))]):
    #                 df=df.replace({0:{'\?':'','\.':''}},regex=True) # Remove ? and .
    #                 # for k_row in df.index: # Remove ? and .
    #                 #     timestr=df.loc[k_row,0]
    #                 #     timestr=timestr.replace('?','')
    #                 #     timestr=timestr.replace('.','')
    #                 #     df.loc[k_row,0]=timestr
    #             try:
    #                 datetime_arr = pd.to_datetime(df["IntD"]+" "+df["IntT"]+" "+df[0],format="%m/%d/%Y %I:%M:%S %p")
    #                 # datetime_arr = pd.to_datetime(df["IntD"] + " " + df["IntT"], format=dateformat, dayfirst=True)
    #                 # try:
    #                 #     datetime_arr[df[df[0] == "PM"].index] = datetime_arr[df[df[0] == "PM"].index] + timedelta(hours=12)
    #                 # except: pass
    #                 # idx = np.argmin(np.diff(datetime_arr))
    #                 # if np.diff(datetime_arr)[idx].astype("float")<0:
    #                 #     datetime_arr[idx+1:] = np.copy(datetime_arr[idx+1:]+timedelta(hours=12))
    #                 arr = list(datetime_arr.values.astype(float) / 10 ** 9)
    #                 if ref_date and abs(arr[0] - ref_date) > 30*24*60*60:
    #                     arr = list(
    #                         datetime_arr.values.astype(float) / 10 ** 9)
    #                 df["time"] = arr
    #                 return df
    #             except:
    #                 datetime_arr = pd.to_datetime(df["IntD"]+" "+df["IntT"]+" "+df[0],format="%d-%b-%y %I:%M:%S %p")
    #                 # datetime_arr = pd.to_datetime(df["IntD"] + " " + df["IntT"], format="%d-%b-%y %H:%M:%S")
    #                 # try:
    #                 #     datetime_arr[df[df[0] == "PM"].index] = datetime_arr[df[df[0] == "PM"].index] + timedelta(
    #                 #         hours=12)
    #                 # except:
    #                 #     pass
    #                 # idx = np.argmin(np.diff(datetime_arr))
    #                 # if np.diff(datetime_arr)[idx].astype("float") < 0:
    #                 #     datetime_arr[idx + 1:] = np.copy(datetime_arr[idx + 1:] + timedelta(hours=12))
    #                 arr = list(datetime_arr.values.astype(float) / 10 ** 9)
    #                 if ref_date and abs(arr[0] - ref_date) > 30 * 24 * 60 * 60:
    #                     arr = list(
    #                         datetime_arr.values.astype(float) / 10 ** 9)
    #                 df["time"] = arr
    #                 return df

    #         else:
    #             del columns[-1]
    #             columns.insert(columns.index("IntT")+1, 0)
    #             df.columns = columns
    #             units.insert(columns.index(0), 0)
    #             if "?" in str([ele for ele in AM_PM if(ele in list(df[0]))]):
    #                 df=df.replace({0:{'\?':'','\.':''}},regex=True) # Remove ? and .
    #                 # for k_row in df.index: # Remove ? and .
    #                 #     timestr=df.loc[k_row,0]
    #                 #     timestr=timestr.replace('?','')
    #                 #     timestr=timestr.replace('.','')
    #                 #     df.loc[k_row,0]=timestr
    #             try:
    #                 datetime_arr = pd.to_datetime(df["IntD"]+" "+df["IntT"]+" "+df[0],format="%m/%d/%Y %I:%M:%S %p")
    #                 # datetime_arr = pd.to_datetime(df["IntD"] + " " + df["IntT"], format=dateformat, dayfirst=True)
    #                 # try:
    #                 #     datetime_arr[df[df[0] == "PM"].index] = datetime_arr[df[df[0] == "PM"].index] + timedelta(hours=12) 
    #                 # except: pass
    #                 # idx = np.argmin(np.diff(datetime_arr))
    #                 # if np.diff(datetime_arr)[idx].astype("float")<0:
    #                 #     datetime_arr[idx+1:] = np.copy(datetime_arr[idx+1:]+timedelta(hours=12))
    #                 arr = list(datetime_arr.values.astype(float) / 10 ** 9)
    #                 if ref_date and abs(arr[0] - ref_date) > 30*24*60*60:
    #                     arr = list(
    #                         datetime_arr.values.astype(float) / 10 ** 9)
    #                 df["time"] = arr
    #                 return df
    #             except:
    #                 log("Datetime file parse failed")
    #                 raise
        
    #     if "IntDT" in columns and "IntDT1" in columns:
    #         if bool([ele for ele in AM_PM if(ele in list(df["IntDT1"]))])==True:
    #             del columns[-1]
    #             columns.insert(columns.index("IntDT1"), 0) 
    #             df.columns=columns
    #             if "?" in str([ele for ele in AM_PM if(ele in list(df["IntDT1"]))]):
    #                 df=df.replace({0:{'\?':'','\.':''}},regex=True) # Remove ? and .
    #                 # for k_row in df.index: # Remove ? and .
    #                 #     timestr=df.loc[k_row,0]
    #                 #     timestr=timestr.replace('?','')
    #                 #     timestr=timestr.replace('.','')
    #                 #     df.loc[k_row,0]=timestr
    #             try:
    #                 datetime_arr = pd.to_datetime(df["IntDT1"]+" "+df["IntDT"]+" "+df[0],format="%m/%d/%Y %I:%M:%S %p")
    #                 # datetime_arr = pd.to_datetime(df["IntDT1"] + " " + df["IntDT"], format=dateformat, dayfirst=True)
    #                 # try:
    #                 #     datetime_arr[df[df[0] == "PM"].index] = datetime_arr[df[df[0] == "PM"].index] + timedelta(hours=12) 
    #                 # except: pass
    #                 # idx = np.argmin(np.diff(datetime_arr))
    #                 # if np.diff(datetime_arr)[idx].astype("float")<0:
    #                 #     datetime_arr[idx+1:] = np.copy(datetime_arr[idx+1:]+timedelta(hours=12))
    #                 arr = list(datetime_arr.values.astype(float) / 10 ** 9)
    #                 if ref_date and abs(arr[0] - ref_date) > 30*24*60*60:
    #                     arr = list(
    #                         datetime_arr.values.astype(float) / 10 ** 9)
    #                 df["time"] = arr
    #                 return df
    #             except:
    #                 log("Datetime file parse failed")
    #                 raise
    #         if bool([ele for ele in AM_PM if(ele in list(df[0]))])==True:
    #             if "?" in str([ele for ele in AM_PM if(ele in list(df[0]))]):
    #                 df=df.replace({0:{'\?':'','\.':''}},regex=True) # Remove ? and .
    #                 # for k_row in df.index: # Remove ? and .
    #                 #     timestr=df.loc[k_row,0]
    #                 #     timestr=timestr.replace('?','')
    #                 #     timestr=timestr.replace('.','')
    #                 #     df.loc[k_row,0]=timestr
    #             try:
    #                 datetime_arr = pd.to_datetime(df["IntDT"]+" "+df["IntDT1"]+" "+df[0],format="%m/%d/%Y %I:%M:%S %p")
    #                 # datetime_arr = pd.to_datetime(df["IntDT"] + " " + df["IntDT1"], format=dateformat, dayfirst=True)
    #                 # try:
    #                 #     datetime_arr[df[df[0] == "PM"].index] = datetime_arr[df[df[0] == "PM"].index] + timedelta(hours=12) 
    #                 # except: pass
    #                 # idx = np.argmin(np.diff(datetime_arr))
    #                 # if np.diff(datetime_arr)[idx].astype("float")<0: #Negative time due to change of day?
    #                 #     datetime_arr[idx+1:] = np.copy(datetime_arr[idx+1:]+timedelta(hours=12))
    #                 arr = list(datetime_arr.values.astype(float) / 10 ** 9)
    #                 if ref_date and abs(arr[0] - ref_date) > 30*24*60*60:
    #                     arr = list(
    #                         datetime_arr.values.astype(float) / 10 ** 9)
    #                 df["time"] = arr
    #                 return df
    #             except:
    #                 log("Datetime file parse failed")
    #                 raise
    #         else:
    #             del columns[-1]
    #             columns.insert(columns.index("IntDT1")+1, 0)
    #             df.columns = columns
    #             units.insert(columns.index(0), 0) #adjusting units
    #             if "?" in str([ele for ele in AM_PM if(ele in list(df[0]))]):
    #                 df=df.replace({0:{'\?':'','\.':''}},regex=True) # Remove ? and .
    #                 # for k_row in df.index: # Remove ? and .
    #                 #     timestr=df.loc[k_row,0]
    #                 #     timestr=timestr.replace('?','')
    #                 #     timestr=timestr.replace('.','')
    #                 #     df.loc[k_row,0]=timestr
    #             try:
    #                 try:
    #                     datetime_arr = pd.to_datetime(df["IntDT"]+" "+df["IntDT1"]+" "+df[0],format="%m/%d/%Y %I:%M:%S %p")
    #                     #datetime_arr = pd.to_datetime(df["IntDT"] + " " + df["IntDT1"], format=dateformat, dayfirst=True)
    #                 except:
    #                     datetime_arr = pd.to_datetime(df["IntDT"]+" "+df["IntDT1"]+" "+df[0],format="%d-%b-%y %I:%M:%S %p")
    #                     # datetime_arr = pd.to_datetime(df["IntDT"] + " " + df["IntDT1"], format="%d-%b-%y %H:%M:%S",
    #                                                   # dayfirst=True)
    #                 # try:
    #                 #     datetime_arr[df[df[0] == "PM"].index] = datetime_arr[df[df[0] == "PM"].index] + timedelta(hours=12) 
    #                 # except: pass
    #                 # idx = np.argmin(np.diff(datetime_arr))
    #                 # if np.diff(datetime_arr)[idx].astype("float")<0:
    #                 #     datetime_arr[idx+1:] = np.copy(datetime_arr[idx+1:]+timedelta(hours=12))
    #                 arr = list(datetime_arr.values.astype(float) / 10 ** 9)
    #                 if ref_date and abs(arr[0] - ref_date) > 30*24*60*60:
    #                     arr = list(
    #                         datetime_arr.values.astype(float) / 10 ** 9)
    #                 df["time"] = arr
    #                 return df
    #             except:
    #                 log("Datetime file parse failed")
    #                 raise              
        
    # else: 
    #     if "IntDT" in columns and "IntDT1" in columns:
    #         try:
    #             arr = list(pd.to_datetime(df["IntDT"] + " " + df["IntDT1"], dayfirst=True).values.astype(float) / 10 ** 9)
    #             if ref_date and abs(arr[0] - ref_date) > 30*24*60*60:
    #                 arr = list(pd.to_datetime(df["IntDT"] + " " + df["IntDT1"], dayfirst=False).values.astype(float) / 10 ** 9)
    #             df["time"] = arr
    #             return df
    #         except:
    #             log("Datetime file parse failed")
    #             raise    
    #     elif "IntD" in columns and "IntT" in columns:
    #         try:
    #             arr = list(pd.to_datetime(df["IntD"] + " " + df["IntT"], dayfirst=True).values.astype(float) / 10 ** 9)
    #             if ref_date and abs(arr[0] - ref_date) > 30*24*60*60:
    #                 arr = list(pd.to_datetime(df["IntD"] + " " + df["IntT"], dayfirst=False).values.astype(float) / 10 ** 9)
    #             df["time"] = arr
    #             return df
    #         except:
    #             log("Datetime file parse failed")
    #             raise
    #     elif "IntD" in columns and "IntD1" in columns:
    #         try:
    #             datetime_arr = pd.to_datetime(df["IntD"] + " " + df["IntD1"], format="%H:%M:%S %m/%d/%Y").values.astype(float) / 10 ** 9
    #             df["time"] = datetime_arr
    #             return df
    #         except:
    #             log("Datetime file parse failed")
    #             raise
    #     elif "IntT" in columns and "IntT1" in columns:
    #         try:
    #             datetime_arr = pd.to_datetime(df["IntT"] + " " + df["IntT1"], format="%H:%M:%S %m/%d/%Y").values.astype(float) / 10 ** 9
    #             df["time"] = datetime_arr
    #             return df
    #         except:
    #             log("Datetime file parse failed")
    #             raise
    #     else:
    #         raise ValueError("Cannot process unrecognised file.")
    

    

def parse_chl(df, variable, name, columns, units, ref_date, date_format):
    if units == "g/l":
        try:
            log("Changed Chl unit")
            return list(df[name] * 1000000)
        except:
            return [-999.] * len(df)
        
    else:
        return [-999.] * len(df)

def qa_std_moving(variable, xdata=np.array([]), window_size=15, factor=3, prior_flags=False):
   """
   Indicate outliers values based on std applied to moving average.
   Parameters:
       variable (np.array): Data array to which to apply the quality assurance
       xdata (np.array): x-values used to resample the data (if not specified, data is not resample)
       window_size (np.int): window size of data
       factor (int): number n such that values higher than n*std are considered as outliers
       prior_flags (np.array): An array of bools where True means non-trusted data
   Returns:
       flags (np.array): An array of bools where True means non-trusted data for this outlier dectection
   """
   if isinstance(prior_flags,np.ndarray): # Boolean array provided
       flags = prior_flags
   else: # No boolean array provided
       flags=np.full(variable.shape,False)

   if len(variable) < window_size:
       print("ERROR! Window size is larger than array length.")
   else:
        # if ~np.any(xdata):
        #     xdata:np.arange(len(variable))
        #Interpolate data at constant intervals
        # xinterp=np.linspace(np.min(xdata),np.max(xdata),len(variable))
        # indsort=np.argsort(xdata)
        # yinterp=np.interp(xinterp, xdata[indsort], variable[indsort])
        # movmean=np.interp(xdata,xinterp,uniform_filter1d(yinterp,size=window_size))
        movmean=uniform_filter1d(variable,size=window_size)
        noise_data=abs(variable-movmean)
        mask_std=noise_data>factor*np.std(noise_data)
        flags=np.logical_or(flags,mask_std)
   return flags

def get_nc_data(nc):
    """
    Get the data from a netCDF file.
    Inputs:
        nc: netCDF object
    Outputs:
        data_nc (dictionary): data for each variable of the netCDF file.
    """
    
    varnames=list(nc.variables)
    data_nc=dict()
    
    for key in varnames:
        data_nc[key]=nc.variables[key][:].data
    return data_nc