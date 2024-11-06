"""
Functions used to make the phosphorus budget of a lake.

Author: T. Doda, Surface Waters - Research and Management, Eawag
Contact: tomy.doda@gmail.com
Date: 17.04.2024
"""

import os
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone, UTC
from scipy.stats import linregress


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_river_load_from_obs(tnum,Qval,TPval,tbudget, calculation="interp"):
    """Function compute_river_load_from_obs

    Calculates the input (or output) of phosphorus from measurements of discharge and TP in rivers and point sources.

    Inputs:
        tnum (numpy array (n,) of floats): time as timestamp values (number of seconds since 01.01.1970)
        Qval (numpy array (n,m) of floats): discharge in each of the m inflows as a function of time [m3.s-1]
        TPval (numpy array (n,m) of floats): TP concentrations in each of the m inflows as a function of time [mg.m-3]
        tbudget (numpy array (p,) of floats): time at which Pin must be computed, as timestamp values (number of seconds since 01.01.1970)
        calculation (string): "interp" or "average"
        
    Outputs:
        Pin_budget (numpy array (p,) of floats): total incoming (outgoing) P load as a function of time [tons-P.yr-1]
        Qin_budget (numpy array (q,) of floats): total incoming (outgoing) discharge as a function of time [m-3.s-1]
        tbudget_new (numpy array (p,) of floats): time at which Pin is computed (differs from tbudget if average calculation is used), as timestamp values (number of seconds since 01.01.1970)
    """
    
    Pin=np.nansum(Qval*TPval,axis=1)*86400*365*1e-9
    
    if calculation=="interp":
        Pin_budget=np.interp(tbudget,tnum,Pin,left=np.nan,right=np.nan) # Linear interpolation
        Qin_budget=np.interp(tbudget,tnum,np.nansum(Qval,axis=1)) # Linear interpolation
        tbudget_new=tbudget
    elif calculation=="average":
        tbudget_new,Pin_budget,_=average_between(tbudget,tnum,Pin) # Average between dates
        _,Qin_budget,_=average_between(tbudget,tnum,np.nansum(Qval,axis=1))
        Pin_budget=np.concatenate((Pin_budget,np.array([np.nan])),axis=0) # Add a nan value at the end
        tbudget_new=np.concatenate((tbudget_new,np.array([np.nan])),axis=0) # Add a nan value at the end      
        Qin_budget=np.concatenate((Qin_budget,np.array([np.nan])),axis=0) # Add a nan value at the end
        
    return Pin_budget, Qin_budget, tbudget_new



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def average_between(xnew,xdata,ydata):
    """
    Computes the average between each x point
    
    Inputs:
        xnew (numpy array (m,) of floats): new x values defining the intervals for which the y data must be averaged
        xdata (numpy array (n,) of floats): initial x values corresponding to the y data
        ydata (numpy array (n,) of floats): y values to average between each xnew value
        
    Outputs:
        xmean (numpy array (m,) of floats): mid-interval values from xnew array
        ymean (numpy array (m,) of floats): averaged y values over each interval from xnew
        ystd (numpy array (m,) of floats): standard deviation for each of the averages ymean
    
    """
    xmean=xnew[:-1]+0.5*(xnew[1:]-xnew[:-1])
    ymean=np.full(len(xnew)-1,np.nan)
    ystd=np.full(len(xnew)-1,np.nan)
    for k in range(len(xmean)):
        bool_avg=np.logical_and(xdata>=xnew[k],xdata<xnew[k+1])
        if np.sum(bool_avg)>0: # At least one value
            ymean[k]=np.nanmean(ydata[bool_avg])
            ystd[k]=np.nanstd(ydata[bool_avg])
        else:
            print("No value for averaging between {} and {}".format(xnew[k],xnew[k+1]))
    
    return xmean,ymean, ystd
    
    

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_river_load_from_curve(Qcurve,TPcurve,tnum,Qval,tbudget,method="power",calculation="interp"):
    """Function compute_river_load_from_curve

    Calculates the input (or output) of phosphorus from a Q-TP relationship applied to discharge values.

    Inputs:
        Qcurve (numpy array (n,m) of floats): discharge in each of the m inflows as a function of time [m3.s-1]
        TPcurve (numpy array (n,m) of floats): TP concentrations in each of the m inflows as a function of time [mg.m-3]
        tnum (numpy array (p,) of floats): time as timestamp values for which TPin must be estimated (number of seconds since 01.01.1970)
        Qval (numpy array (p,m) of floats): discharge in each of the m inflows as a function of time for which TPin must be estimated [m3.s-1]
        tbudget (numpy array (q,) of floats): time at which Pin must be computed, as timestamp values (number of seconds since 01.01.1970)
        method (string): regression method, the only option is currently "power": TP=a*Q^b (other options can be added)
        calculation (string): "interp" or "average"
             
    Outputs:
        Pin_budget (numpy array (q,) of floats): total incoming (outgoing) P load as a function of time [tons-P.yr-1]
        Qin_budget (numpy array (q,) of floats): total incoming (outgoing) discharge as a function of time [m-3.s-1]
        tbudget_new (numpy array (q,) of floats): time at which Pin is computed (differs from tbudget if average calculation is used), as timestamp values (number of seconds since 01.01.1970)
        param (numpy array (x,m) of floats): x parameters of the regression curve for each inflow (e.g., a and b for "power")
        R2 (numpy array (m,) of floats): R2 value of the regression curve for each inflow 
    """
    # Compute curve
    TPval=np.full(Qval.shape,np.nan)
    R2=np.full((Qval.shape[1],),np.nan)
    if method=="power":
        param=np.full((2,Qval.shape[1]),np.nan)
        for kin in range(Qcurve.shape[1]):
            bool_keep=np.logical_and(Qcurve[:,kin]>0,TPcurve[:,kin]>0)
            regres= linregress(np.log(Qcurve[bool_keep,kin]),np.log(TPcurve[bool_keep,kin]))
            param_log=np.array([regres.slope,regres.intercept])
            param[:,kin]=[np.exp(param_log[1]),param_log[0]]
            R2[kin]=regres.rvalue**2
            TPval[Qval[:,kin]>0,kin]=np.exp(np.polyval(param_log,np.log(Qval[Qval[:,kin]>0,kin])))
    Pin=np.nansum(Qval*TPval,axis=1)*86400*365*1e-9 
    if calculation=="interp":
        Pin_budget=np.interp(tbudget,tnum,Pin) # Linear interpolation
        Qin_budget=np.interp(tbudget,tnum,np.nansum(Qval,axis=1)) # Linear interpolation
        tbudget_new=tbudget
    elif calculation=="average":
        tbudget_new,Pin_budget,_=average_between(tbudget,tnum,Pin) # Average between dates
        _,Qin_budget,_=average_between(tbudget,tnum,np.nansum(Qval,axis=1))
        Pin_budget=np.concatenate((Pin_budget,np.array([np.nan])),axis=0) # Add a nan value at the end
        Qin_budget=np.concatenate((Qin_budget,np.array([np.nan])),axis=0) # Add a nan value at the end
        tbudget_new=np.concatenate((tbudget_new,np.array([np.nan])),axis=0) # Add a nan value at the end
    
    return Pin_budget, Qin_budget, tbudget_new, param, R2

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def discharge_TP(tnum,Qin):
    """Function discharge_TP

    Power function TPin=a*Qin^b providing the TP concentration in the inflow as a function of the inflow discharge (could be modified to use other functions).

    Inputs:
        tnum (numpy array (n,) of floats): time as timestamp values for which Qin is measured (number of seconds since 01.01.1970)
        Qin (numpy array (n,) of floats): inflow discharge [m3.s-1]
             
    Outputs:
        TPin (numpy array (n,) of floats): TP concentration in the inflow [mg.m-3] 
    """
    
    # Function provided for Baldeggersee by Müller et al. (2012)
    TPin=np.full(Qin.shape,np.nan)
    period1_logical=tnum<datetime(1996,1,1).replace(tzinfo=timezone.utc).timestamp()
    TPin[period1_logical]=100.6*(Qin[period1_logical]*86400*365/1e6)**0.037
    TPin[~period1_logical]=18.7*(Qin[~period1_logical]*86400*365/1e6)**0.41
    
    return TPin



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_TP_2boxes(depthval,TPval,hepi):
    """Function compute_TP_2boxes

    Calculates the depth-averaged TP concentration in the epilimnion and in the hypolimnion.

    Inputs:
        depthval (numpy array (n,) of floats): depth values [m]
        TPval (numpy array (n,m) of floats): TP concentrations as a function of depth and time [mg.m-3]
        hepi (numpy array (m,) of floats): thermocline depth as a function of time, hepi=0 when complete mixing [m]
            
    Outputs:
        TPepi (numpy array (m,) of floats): depth-averaged TP concentration in the epilimnion [mg.m-3]
        TPhypo (numpy array (m,) of floats): depth-averaged TP concentration in the hypolimnion [mg.m-3]
    """
    TPepi=np.full(len(hepi),np.nan)
    TPhypo=np.full(len(hepi),np.nan)
    
    for kt in range(len(hepi)):
        # Calculation of TPhypo
        bool_hypo=depthval>=hepi[kt]
        bool_epi=depthval<hepi[kt]
        if np.nansum(bool_hypo)>0: # At least one value in the hypolimnion
            TPhypo[kt]=np.nanmean(TPval[bool_hypo,kt])
        if np.nansum(bool_epi)>0: # At least one value in the epilimnion
            TPepi[kt]=np.nanmean(TPval[bool_epi,kt])
        else:
            TPepi[kt]=TPhypo[kt]
        if np.nansum(bool_hypo)==0:
            TPhypo[kt]=TPepi[kt]
    
    return TPepi, TPhypo

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def extract_simstrat_T(filename):
    """Function extract_simstrat_T

    Extracts the data from a temperature Simstrat output file (".dat").

    Inputs:
        filename (string): path to the Simstrat file.
        
    Outputs:
        tnum (numpy array (n,) of floats): time as timestamp values (number of seconds since 01.01.1970)
        depthval (numpy array (m,) of floats): depth [m]
        tempval (numpy array (m,n) of floats): water temperature [°C]
    """
    df_stratif=pd.read_csv(filename, sep=",",header=0)
    daynb=df_stratif["Datetime"].to_numpy()
    tnum=datetime(1981,1,1).replace(tzinfo=timezone.utc).timestamp()+daynb*86400
    depthval=df_stratif.columns[1:].to_numpy().astype(float)
    tempval=df_stratif.values[:,1:].astype(float)
    
    # Transpose matrix to have rows=depths, columns=time
    tempval=tempval.transpose()
    
    # Reorder depths as positive values from surface to bottom
    depthval=np.sort(-depthval)
    tempval=np.flipud(tempval)
    
    return tnum, depthval, tempval


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_anoxia_dates(datesval,tdate_start,hepi,zmax):
    """Function compute_hepi_constant

    Determines the anoxic period based as the period between the specified starting date and the end of the stratified period.

    Inputs:
        datesval (numpy array (n,) of datetime): dates for which the thermocline depth must be computed
        tdate_start (datetime): starting date of the anoxic period every year (dd.mm.1900) 
        hepi (numpy array (n,) of floats): thermocline depth [m]
        zmax (float): lake maximum depth [m]
        
    Outputs:
        bool_anox (numpy array (n,) of booleans): = True if anoxic, = False otherwise
    """
    bool_anox=np.full(len(datesval),False)
    anox_before=False
    for kt in range(len(datesval)):
        if hepi[kt]<zmax: # stratified period
            if not anox_before and datesval[kt]>datetime(datesval[kt].year,tdate_start.month,tdate_start.day): # Start of anoxic period
                bool_anox[kt]=True
                anox_before=True
            elif anox_before: # already anoxic before
                bool_anox[kt]=True
        elif anox_before: # end of the anoxic period
            anox_before=False
    
    return bool_anox
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_hepi_constant(datesval,tdate_start,tdate_end,hepi,zmax,filter_stratif=np.array([])):
    """Function compute_hepi_constant

    Computes time series of thermocline depths based on the stratification period and a constant thermocline depth.

    Inputs:
        datesval (numpy array (n,) of datetime): dates for which the thermocline depth must be computed
        tdate_start (datetime): starting date of the stratified period (dd.mm.1900)
        tdate_end (datetime): end date of the stratified period (dd.mm.1900)
        hepi (float): thermocline depth [m]
        zmax (float): lake maximum depth [m]
        filter_stratif (numpy array (n,) of booleans): additional filter to apply on the data, =False to prevent stratified conditions between the start and ending dates
        
    Outputs:
        hepi_val: time series of thermocline depths [m]
    """
    hepi_val=np.full(len(datesval),np.nan)
    if not list(filter_stratif): # Empty
        filter_stratif=np.full(len(datesval),True)
    for kt in range(len(datesval)):
        if datesval[kt]>datetime(datesval[kt].year,tdate_start.month,tdate_start.day) and \
        datesval[kt]<datetime(datesval[kt].year,tdate_end.month,tdate_end.day): # Startified period
            hepi_val[kt]=hepi
        else:
            hepi_val[kt]=zmax
    
    return hepi_val


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def movmean(X,windowsize,axis=0):
    """Function movmean

    Computes the moving average of an array centered at the given index.

    Inputs:
        X (numpy array (m,n) of floats): array to average
        windowsize (int): size of the averaging window
        axis (int): index of the axis along which the averaging is applied
        
    Outputs:
        X_smooth (numpy array (m,n) of floats): smoothed array
    """

    if len(X.shape)==1:
        X=np.expand_dims(X,axis=1)
    if axis==1:
        X=X.transpose()
    X_smooth=np.full(X.shape,np.nan)
    for k in range(X.shape[1]): 
        df=pd.DataFrame({'val':X[:,k]})
        X_smooth[:,k]=df.rolling(windowsize,center=True).mean().values[:,0]
    if axis==1:
        X_smooth=X_smooth.transpose()
    return X_smooth

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_hepi_varying(tnum_TP,tnum_T,depth_T,tempval,zmax,mingrad=0.05,windowsize=4):
    """Function compute_hepi_varying

    Computes time series of thermocline depths based on temperature data by calculating the depth of the maximal temperature gradient.

    Inputs:
        tnum_TP (numpy array (p,) of floats): timestamps for which the thermocline depth must be computed (number of seconds since 01.01.1970)
        tnum_T (numpy array (n,) of floats): timestamps of the temperature data (number of seconds since 01.01.1970)
        depth_T (numpy array (m,) of floats): depth values of the temperature data [m]
        tempval (numpy array (m,n) of floats): water temperature as a function of depth and time [°C]
        zmax (float): lake maximum depth [m]
        mingrad (float): minimum temperature gradient below which the water column is considered fully mixed [°C/m]
        windowsize (int): size of the averaging window for temperature vertical smoothing 
        
    Outputs:
        hepi_val: time series of thermocline depths at time steps of interest [m]
        hepi_all: time series of thermocline depths at time steps of temperature measurements [m]
    """
    hepi_all=np.full(len(tnum_T),np.nan)
    
    # Compute the thermocline depth for each day with temperature data
    temp_smooth=movmean(tempval,windowsize)
    depth_T=np.expand_dims(depth_T,axis=1) # (n,1) array
    grad_temp=np.abs((temp_smooth[1:,:]-temp_smooth[:-1,:])/(depth_T[1:]-depth_T[:-1]))
    bool_mixed=np.nanmax(grad_temp,axis=0)<mingrad
    hepi_all[bool_mixed]=zmax
    hepi_all[~bool_mixed]=depth_T[np.nanargmax(grad_temp[:,~bool_mixed],axis=0)].transpose()[0]
    
    # Linear interpolation to selected time steps:
    hepi_val=np.interp(tnum_TP,tnum_T,hepi_all,left=np.nan,right=np.nan)
      
    
    return hepi_val, hepi_all

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_remobilization_Hanson(T_hypo,TP_sed,A_therm,a=-4.3,b=22.86,theta=1.172,Tbase=10):
    """Function compute_remobilization_Hanson

    Computes the phosphorus remobilization flux based on Hanson et al. (2020). 

    Inputs:
        T_hypo (numpy array (n,) of floats): hypolimnion temperature [°C]
        TP_sed (float): phosphorus concentration at the sediment surface [mg-P/g-sed]
        A_therm (numpy array (n,) of floats): surface area at the thermocline depth [m2]
        a, b, theta, Tbase (float): empirical coefficients from Hanson et al. (2020)
        
    Outputs:
        P_remob (numpy array (n,) of floats): phosphorus remobilization mass flux [tons-P/yr]
    """
    CF=a+b*TP_sed
    P_remob=A_therm*CF*theta**(T_hypo-Tbase)*365*1e-9 # [tons-P/yr] 
    # Note: P_remob=0 when A_therm==0 (fully mixed, no hypoxia)
      
    return P_remob
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_remobilization_Nurnberg(TP_sed,A_sed,a=0.8,b=0.76):
    """Function compute_remobilization_Nurnberg

    Computes the phosphorus remobilization flux based on Nürnberg (1988). 

    Inputs:
        TP_sed (float): phosphorus concentration at the sediment surface [mg-P/g-sed]
        A_sed (numpy array (n,) of floats): sediment surface area below the thermocline [m2]
        a, b (float): empirical coefficients from Nürnberg et al. (1988)
        
    Outputs:
        P_remob (numpy array (n,) of floats): phosphorus remobilization mass flux [tons-P/yr]
    """
    P_remob=np.exp(a+b*np.log(TP_sed))*A_sed*365*1e-9 # [tons-P/yr]
    # Note: P_remob=0 when A_sed==0 (fully mixed, no hypoxia)
      
    return P_remob

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_remobilization_Carter(TP_epi,A_sed,a=-0.54,b=0.827,avg=False):
    """Function compute_remobilization_Carter

    Computes the phosphorus remobilization flux based on Carter & Dzialowski (2012). 

    Inputs:
        TP_epi (numpy array (n,) of floats): TP concentration in the epilimnion [mg-P/m3]
        A_sed (numpy array (n,) of floats): sediment surface area below the thermocline [m2]
        a, b (float): empirical coefficients from Carter & Dzialowski (2012)
        avg (boolean): if =True, use the average TP_epi during stratifcation period
        
    Outputs:
        P_remob (numpy array (n,) of floats): phosphorus remobilization mass flux [tons-P/yr]
    """
    P_remob=np.full(TP_epi.shape,0) # Set P_remob to zero if TP_epi==0
    if not isinstance(TP_epi,np.ndarray):
        if TP_epi>0:
            P_remob=np.exp(a+b*np.log(TP_epi))*A_sed*365*1e-9 # [tons-P/yr]
    else:        
        ind_stratif=np.where(TP_epi>0)[0]
        ind_stratif_start=np.concatenate((np.array([ind_stratif[0]]),ind_stratif[np.where(np.diff(ind_stratif)>1)[0]+1]))
        ind_stratif_end=np.concatenate((ind_stratif[np.where(np.diff(ind_stratif)>1)[0]-1],np.array([ind_stratif[-1]])))
       
        for kstrat in range(len(ind_stratif_start)): # Each stratified period: compute P_remob 
            if avg:
                TPavg=np.nanmean(TP_epi[ind_stratif_start[kstrat]:ind_stratif_end[kstrat]+1])
                P_remob[ind_stratif_start[kstrat]:ind_stratif_end[kstrat]+1]=np.exp(a+b*np.log(TPavg))*A_sed[ind_stratif_start[kstrat]:ind_stratif_end[kstrat]+1]*365*1e-9 # [tons-P/yr]
            else:
                P_remob[ind_stratif_start[kstrat]:ind_stratif_end[kstrat]+1]=np.exp(a+b*np.log(TP_epi[ind_stratif_start[kstrat]:ind_stratif_end[kstrat]+1]))*A_sed[ind_stratif_start[kstrat]:ind_stratif_end[kstrat]+1]*365*1e-9 # [tons-P/yr]
            # Note: P_remob=0 when A_sed==0 (fully mixed, no hypoxia)
          
    return P_remob

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_remobilization_hypo(tnum_budget,tnum_TP,TPhypo,Vhypo,hepi_budget,hmax):
    """Function compute_remobilization_hypo

    Computes the phosphorus remobilization flux based on changes in hypolimnetic phosphorus content.

    Inputs:
        tnum_budget (numpy array (n,) of floats): timestamps when Premob must be calculated (number of seconds since 01.01.1970)
        tnum_TP (numpy array (m,) of floats): timestamps when TP is measured (number of seconds since 01.01.1970)
        TPhypo (numpy array (m,) of floats): TP concentration in the hypolimnion [mg-P/m3]
        Vhypo (numpy array (m,) of floats): volume of the hypolimnion [m3]
        hepi_budget (numpy array (n,) of floats): epilimnion thickness [m]
        hmax (float): maximum lake depth
         
    Outputs:
        P_remob (numpy array (n,) of floats): phosphorus remobilization mass flux [tons-P/yr]
    """
    P_remob=np.full(len(tnum_budget),0)
    ind_stratif=np.where(hepi_budget<hmax)[0]
    ind_stratif_start=np.concatenate((np.array([ind_stratif[0]]),ind_stratif[np.where(np.diff(ind_stratif)>1)[0]+1]))
    ind_stratif_end=np.concatenate((ind_stratif[np.where(np.diff(ind_stratif)>1)[0]-1],np.array([ind_stratif[-1]])))
    
    for kstrat in range(len(ind_stratif_start)): # Each stratified period: compute P_remob
        ind_TP=np.where(np.logical_and(tnum_TP>=tnum_budget[ind_stratif_start[kstrat]],tnum_TP<=tnum_budget[ind_stratif_end[kstrat]]))[0]
        if ind_TP.size>0:
            indmin=np.argmin(TPhypo[ind_TP])
            indmax=np.argmax(TPhypo[ind_TP])
            Vhypo_mean=np.nanmean(Vhypo[ind_TP[Vhypo[ind_TP]>0]])
            if indmin<indmax:
                #param=np.polyfit(tnum_TP[ind_TP[indmin:indmax+1]]/(86400*365),TPhypo[ind_TP[indmin:indmax+1]]*Vhypo[ind_TP[indmin:indmax+1]]*1e-9,1)
                #P_remob[ind_stratif_start[kstrat]:ind_stratif_end[kstrat]+1]=param[0]
                P_remob[ind_stratif_start[kstrat]:ind_stratif_end[kstrat]+1]=Vhypo_mean*(TPhypo[ind_TP[indmax]]-TPhypo[ind_TP[indmin]])/(tnum_TP[ind_TP[indmax]]-tnum_TP[ind_TP[indmin]])*86400*365*1e-9
    P_remob[P_remob<0]=0
    
    return P_remob

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_changes_TP(tnum_TP,TP_epi,TP_hypo,tnum_budget):
    """Function compute_changes_TP

    Computes the temporal changes in TP at the time steps specified for the P budget. 

    Inputs:
        tdate_TP (numpy array (n,) of floats): timestamps with TP data (number of seconds since 01.01.1970)
        TP_epi (numpy array (n,) of floats): TP concentration in the epilimnion [mg-P/m3]
        TP_hypo (numpy array (n,) of floats): TP concentration in the hypolimnion [mg-P/m3]
        tnum_budget (numpy array (m,) of floats): timestamps when TP changes must be calculated (number of seconds since 01.01.1970)
        
    Outputs:
        dTPepi_dt (numpy array (m,) of floats): temporal changes in epilimnetic TP [mg-P.m-3.s-1]
        dTPhypo_dt (numpy array (m,) of floats): temporal changes in hypolimnetic TP [mg-P.m-3.s-1]
    """
    if tnum_TP[0]>=tnum_budget[0] or tnum_TP[-1]<=tnum_budget[-1]:
        raise Exception("TP measurements are needed before and after the budget period")
        
    dTPepi_dt=np.full(len(tnum_budget),np.nan)
    dTPhypo_dt=np.full(len(tnum_budget),np.nan)
    
    for kt in range (len(tnum_budget)):
        ind_before=np.where(tnum_TP<tnum_budget[kt])[0][-1]
        ind_after=np.where(tnum_TP>tnum_budget[kt])[0][0]
        dTPepi_dt[kt]=(TP_epi[ind_after]-TP_epi[ind_before])/(tnum_TP[ind_after]-tnum_TP[ind_before]) # [mg-P.m-3.s-1]
        dTPhypo_dt[kt]=(TP_hypo[ind_after]-TP_hypo[ind_before])/(tnum_TP[ind_after]-tnum_TP[ind_before]) # [mg-P.m-3.s-1]
    
    return dTPepi_dt, dTPhypo_dt

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_gross_sed_Hanson(Pin,TP_epi,T_epi,V_epi,C_pp=0.5,C_sed=0.0137,theta_sed=1.065,Tbase_sed=10):
    """Function compute_gross_sed_Hanson

    Computes the phosphorus gross sedimentation flux based on Hanson et al. (2020). 

    Inputs:
        Pin (numpy array (n,) of floats): input phosphorus load [tons-P/yr]
        TP_epi (numpy array (n,) of floats): TP concentration in the epilimnion [mg-P.m-3]
        T_epi (numpy array (n,) of floats): epilimnion temperature [°C]
        V_epi (numpy array (n,) of floats): epilimnion volume [m3]
        C_pp (float): fraction of Pin that is particulate P
        C_sed (float): first-order decay rate for the epilimnetic P pool [day-1]
        theta_sed (float): Arrhenius coefficient for temperature scaling of sedimentation
        Tbase_sed (float): base temperature for sedimentation temperature scaling [°C]
        
    Outputs:
        P_gross (numpy array (n,) of floats): gross phosphorus sedimentation flux [tons-P/yr]
    """
    
    P_gross=Pin*C_pp+V_epi*TP_epi*C_sed*theta_sed**(T_epi-Tbase_sed)*365*1e-9 # [tons-P/yr]
    return P_gross
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_net_sed_Vollenweider(TP,sigma_max,V,P_NS_max=np.nan,TPcrit=np.nan):
    """Function compute_net_sed_Vollenweider

    Computes the net phosphorus sedimentation flux with the Vollenweider method.

    Inputs:
        TP (numpy array (n,) of floats): depth-averaged TP concentrations [mg-P/m3]
        sigma_max (float): maximal net sedimentation rate reached for low TP concentrations (slope of the linear relationship P_NS=f(TP)) [yr-1]
        V (float): lake volume [m3]
        P_NS_max (float): maximal net sedimentation flux, reached for high TP concentrations [tons-P/yr]. If nan, only linear relationship is used.
        TPcrit (float): critical TP concentration above which sigma is not constant [mg-P/m3]
        
    Outputs:
        P_NS (numpy array (n,) of floats): net phosphorus sedimentation flux [tons-P/yr]
        sigma_max (float): maximal net sedimentation rate reached for low TP concentrations (slope of the linear relationship P_NS=f(TP)) [yr-1]
        TPcrit (float): critical TP concentration above which sigma is not constant [mg-P/m3]
    """   
    
    P_NS=np.full(TP.shape,np.nan)
    if not np.isnan(P_NS_max) and not np.isnan(TPcrit): # Compute sigma_max
        sigma_max=P_NS_max*1e9/(TPcrit*V) # [yr-1]
        P_NS[TP<TPcrit]=sigma_max*TP[TP<TPcrit]*V*1e-9 
        P_NS[TP>=TPcrit]=P_NS_max
    elif np.isnan(P_NS_max) and not np.isnan(sigma_max):
        P_NS=sigma_max*TP*V*1e-9 # No saturation of P_NS
    elif not np.isnan(sigma_max):
        TPcrit=P_NS_max*1e9/(sigma_max*V) # TP concentration where the two models meet [mg/m3] 
        P_NS[TP<TPcrit]=sigma_max*TP[TP<TPcrit]*V*1e-9 
        P_NS[TP>=TPcrit]=P_NS_max
    else:
        raise Exception("Not enough parameters were provided")
    
    return P_NS, sigma_max, TPcrit

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_stratified_periods(tnum,hepi,zmax):
    """Function compute_stratified_periods

    Computes the starting and ending time of each stratified periods from the time series of thermocline depths.

    Inputs:
        tnum_budget (numpy array (n,) of floats): timestamps when thermocline depth is known (number of seconds since 01.01.1970)
        hepi (numpy array (n,) of floats): thermocline depth [m]
        zmax (float): maximum lake depth [m]
    Outputs:
        tnum_stratif (numpy array (2,m) of floats): timestamps for start and end of each stratified period (number of seconds since 01.01.1970)
    """  
    
    ind_stratif=np.where(hepi<zmax)[0]
    ind_start=ind_stratif[np.concatenate((np.array([True]),np.diff(ind_stratif)>1))] # First index of each stratified period
    ind_end=np.full(len(ind_start),np.nan)
    tnum_periods=np.full((2,len(ind_start)),np.nan)
    for kt in range(len(ind_start)):
        if kt<len(ind_start)-1:
            ind_end=ind_start[kt]+np.where(hepi[ind_start[kt]:ind_start[kt+1]]<zmax)[0][-1]
        else:
            ind_end=ind_start[kt]+np.where(hepi[ind_start[kt]:]<zmax)[0][-1]
        tnum_periods[:,kt]=np.array([tnum[ind_start[kt]],tnum[ind_end]])

    return tnum_periods

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_period_to_anoxia(z_hypo,Fred=0.36,C0=11,delta=0.82e-3,DO2=1e-4):
    """Function compute_period_to_anoxia

    Computes the duration of the stratified period before the start of anoxia, following Müller et al. (2012).

    Inputs:
        z_hypo (float): average thickness of the hypolimnion [m]
        Fred (float): areal flux of reduced substances to the hypolimnion water [g-O2 m-2 d-1]
        C0 (float): initial O2 concentration in the hypolimnion at the beginning of the stratified period (spring) [mg L-1]
        delta (float): thickness of the diffusive boundary layer [m]
        DO2 (float): molecular O2 diffusion coefficient [m2 d-1]
    Outputs:
        delta_t (float): duration of the period between the start of stratification and anoxic conditions in the hypolimnion [days]
    """  
    delta_t=delta/DO2*z_hypo*np.log(1+DO2*C0/(Fred*delta))
    return delta_t

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_anoxia_red(tnum_budget,hepi_model,z_mean,Fred=0.36,C0=11,delta=0.82e-3,DO2=1e-4,tnum_stratif=np.array([])):
    """Function compute_anoxia_red

    Computes when anoxic conditions occur in the hypolimnion, based on the empirical relationship from Müller et al. (2012).

    Inputs:
        tnum_budget (numpy array (n,) of floats): timestamps when thermocline depth is known (number of seconds since 01.01.1970)
        hepi_model (numpy array (n,) of floats): thermocline depth of the box model (the lake is mixed when the thermocline depth reaches the lake mean depth) [m]
        z_mean (float): mean lake depth [m]
        Fred (float): areal flux of reduced substances to the hypolimnion water [g-O2 m-2 d-1]
        C0 (float): initial O2 concentration in the hypolimnion at the beginning of the stratified period (spring) [mg L-1]
        delta (float): thickness of the diffusive boundary layer [m]
        DO2 (float): molecular O2 diffusion coefficient [m2 d-1]
        tnum_stratif (numpy array (2,m) of floats): timestamps for start and end of each stratified period (number of seconds since 01.01.1970)
    Outputs:
        bool_anoxic (numpy array (n,) of booleans): =True during anoxic period, =False otherwise
        ndays_to_anox (numpy array (p,) of booleans): number of days nefore reaching anoxia for each stratified period
        tstart_anox (numpy array (p,) of booleans): timestamps when anoxic period starts (number of seconds since 01.01.1970)
    """  
    
    bool_anoxic=np.full(len(tnum_budget),False)
    ind_stratif=np.where(hepi_model<z_mean)[0]
    ind_start=ind_stratif[np.concatenate((np.array([True]),np.diff(ind_stratif)>1))] # First index of each stratified period
    ndays_to_anox=np.full(len(ind_start),np.nan)
    tstart_anox=np.full(len(ind_start),np.nan)
    # Compute delta_t anox for each stratified period
    for kp in range(len(ind_start)):
        if kp<len(ind_start)-1:
            ind_periods=np.arange(ind_start[kp],ind_start[kp]+np.where(hepi_model[ind_start[kp]:ind_start[kp+1]]<z_mean)[0][-1]+1,1)
        else:
            ind_periods=np.arange(ind_start[kp],ind_start[kp]+np.where(hepi_model[ind_start[kp]:]<z_mean)[0][-1]+1,1)
        z_hypo_mean=np.nanmean(z_mean-hepi_model[ind_periods]) # Mean thickness of the hypolimnion during the stratified period
        ndays_to_anox[kp]=compute_period_to_anoxia(z_hypo_mean,Fred,C0,delta,DO2) # Time required before reaching anoxia [days]
        if list(tnum_stratif): # Detailed stratified periods are specified
            ind_stratif_period=np.where(np.logical_and(tnum_budget[ind_start[kp]]>=tnum_stratif[0,:],tnum_budget[ind_start[kp]]<tnum_stratif[1,:]))[0][0]
            tstratif_0=tnum_stratif[0,ind_stratif_period]
            tstratif_end=tnum_stratif[1,ind_stratif_period]
        else: # Use the thermocline depth provided
            tstratif_0=tnum_budget[ind_periods[0]]
            tstratif_end=tnum_budget[ind_periods[-1]]
        
        if (tstratif_end-tstratif_0)/86400>ndays_to_anox[kp]: # Stratified period is long enough to reach anoxia
            tstart_anox[kp]=tstratif_0+ndays_to_anox[kp]*86400    
            ind_anox=np.where(tnum_budget>=tstart_anox[kp])[0][0]
            bool_anoxic[ind_anox:ind_periods[-1]+1]=True
 
    return bool_anoxic,ndays_to_anox,tstart_anox

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def predict_TP_model(tnum_rivers,bool_anoxic,Pin,Qout,hepi,TPepi0,TPhypo0,TPsed,z_hypso,A_hypso,Thypo=np.array([]),Tepi=np.array([]),sigma_max=np.nan,
                     P_NS_max=np.nan,TPcrit=np.nan,k_sigma=10,Kz=1e-7,zout=0,method_sed="Vollenweider",method_remob="average",show_output=False):
    """Function predict_TP_model

    Predicts TPepi and TPhypo based on a two-box model. 

    Inputs:
        tnum_rivers (numpy array (n,) of floats): timestamps with river data Pin and Qout (number of seconds since 01.01.1970)
        bool_anoxic (numpy array (n,) of booleans): =True during anoxic period, =False otherwise
        Pin (numpy array (n,) of floats): input P load from rivers computed as Pin=Qin*TPin [tons-P/yr]
        Qout (numpy array (n,) of floats): outflow set equal to Qin by assuming that there is no change of water level [m3/s]
        hepi (numpy array (n,) of floats): real thermocline depth [m]
        TPhypo0 (float): initial TP concentration in the hypolimnion [mg-P/m3]
        TPepi0 (float): initial TP concentration in the epilimnion [mg-P/m3]
        TPsed (float): phosphorus concentration at the sediment surface [mg-P/g-sed] 
        z_hypso (numpy array (m,) of floats): depth values where the lake area is provided [m] 
        A_hypso (numpy array (m,) of floats): lake area at the depth values z_hypso [m2]
        Thypo (numpy array (n,) of floats): hypolimnion temperature [°C], if nan the remobilization rate is not estimated with the Hanson method
        Tepi (numpy array (n,) of floats): epilimnion temperature [°C], if nan the gross sedimentation rate cannot be estimated with the Hanson method
        sigma_max (float): maximal net sedimentation rate reached for low TP concentrations (slope of the linear relationship P_NS=f(TP)) [yr-1]. If nan, estimated from z_mean.
        P_NS_max (float): maximal net sedimentation flux, reached for high TP concentrations [tons-P/yr]. If nan, only linear relationship is used.
        TPcrit (float): critical TP concentration above which sigma is not constant [mg-P/m3]
        k_sigma (float): empirical coefficient to compute sigma_max as sigma_max=k_sigma/z_mean [m.yr-1], common range is 8-16 (Müller et al., 2014) 
        Kz (float): vertical turbulent diffusivity [m2/s] 
        zout (float): depth of the outflow [m]
        method_sed (string): method to compute sedimentation, options are "Hanson" (gross sedimentation) or "Vollenweider" (net sedimentation)
        method_remob (string): method to compute remobilization, options are "Nurnberg", "Hanson", "Carter" or "average"
        show_output (boolean): =True to display sedimentation parameters values at each iteration
    
    Outputs:
        tnum_predict (numpy array (n+1,) of floats): timestamps when TP is computed, including initial values (number of seconds since 01.01.1970)
        TPepi (numpy array (n+1,) of floats): average epilimnetic TP [mg-P.m-3]
        TPepi_range (numpy array (2,n+1) of floats): minimum and maximum epilimnetic TP based on min/max Premob at each time step using different methods [mg-P.m-3]
        TPhypo (numpy array (n+1,) of floats): average hypolimnetic TP [mg-P.m-3]
        TPhypo_range (numpy array (2,n+1) of floats): minimum and maximum hypolimnetic TP based on min/max Premob at each time step using different methods [mg-P.m-3]
        P_fluxes (dictionary of numpy arrays (n,) of floats): phosphorus fluxes Pin, Pout_epi, Pout_hypo, Premob, Pnet_sed, Pz [tons-P.yr-1]
        param (dictionary of (n,) numpy arrays): parameters related to the Vollenweider sedimentation flux (e.g., sigma_max, TPcrit, P_NS_max...)
    """  
    
    # Lake parameters
    A0=A_hypso[z_hypso==0]
    V_hypso=np.concatenate((np.mean([A_hypso[1:],A_hypso[:-1]],axis=0)*(z_hypso[1:]-z_hypso[:-1]),np.array([0]))) # Volume of each layer [m3]
    V_lake=np.nansum(V_hypso) # [m3]
    z_mean=V_lake/A0 # Mean lake depth [m]
    #z_max=np.nanmax(z_hypso) # Maximum lake depth [m]
    
    # Initialization
    Pout_epi=np.full(len(tnum_rivers),np.nan)
    Pout_hypo=np.full(len(tnum_rivers),np.nan)
    Pz=np.full(len(tnum_rivers),np.nan)
    Pnet_sed=np.full(len(tnum_rivers),np.nan)
    Premob=np.full(len(tnum_rivers),np.nan)
    Premob_methods=np.zeros((2,len(tnum_rivers))) # Premob from 2 methods: Carter & Nürnberg
    Premob_range=np.zeros((2,len(tnum_rivers)))
    
    #tnum_predict=np.concatenate((np.array([tnum_rivers[0]-0.5*(tnum_rivers[1]-tnum_rivers[0])]),0.5*(tnum_rivers[:-1]+tnum_rivers[1:]),np.array([tnum_rivers[-1]+0.5*(tnum_rivers[-1]-tnum_rivers[-2])])))
    tnum_predict=np.copy(tnum_rivers)
    TPepi=np.full(len(tnum_predict),np.nan)
    TPepi_range=np.full((2,len(tnum_predict)),np.nan)
    TPhypo=np.full(len(tnum_predict),np.nan)
    TPhypo_range=np.full((2,len(tnum_predict)),np.nan)
    TPepi[0]=TPepi0
    TPhypo[0]=TPhypo0
    TPepi_range[:,0]=np.array([TPepi0,TPepi0])
    TPhypo_range[:,0]=np.array([TPhypo0,TPhypo0])
    
    param={"sigma_max":np.full(len(tnum_rivers),np.nan),
           "TPcrit":np.full(len(tnum_rivers),np.nan),
           "P_NS_max":np.full(len(tnum_rivers),np.nan)}
    
    # Computation (temporal loop)
    for kt in range(len(tnum_rivers)-1):
        delta_t=tnum_predict[kt+1]-tnum_predict[kt] # [s]
        Atherm=A_hypso[np.where(z_hypso>=hepi[kt])[0][0]] 
        Vepi=np.nansum(V_hypso[z_hypso<=hepi[kt]])
        
        
        # 1. Vertical turbulent flux (=0 if fully mixed)
        Pz[kt]=2*Kz*Atherm*(TPhypo[kt]-TPepi[kt])/z_mean*86400*365*1e-9 # [tons-P yr-1]
        
        # 2. Outflow
        if zout<hepi[kt]:
            Pout_epi[kt]=Qout[kt]*TPepi[kt]*86400*365*1e-9 # [tons-P yr-1]
            Pout_hypo[kt]=0
        else:
            Pout_epi[kt]=0
            Pout_hypo[kt]=Qout[kt]*TPhypo[kt]*86400*365*1e-9 # [tons-P yr-1]
  
        
        # 3. Remobilization
        if bool_anoxic[kt]: # Anoxic period
            if hepi[kt]==np.max(z_hypso):
                raise Exception('The lake cannot be anoxic during mixing periods')
            if list(Thypo): # Not empty
                Premob_Hanson=compute_remobilization_Hanson(Thypo[kt],TPsed,Atherm) # [tons-P yr-1]
            else:
                Premob_Hanson=np.nan
            Premob_Nurnberg=compute_remobilization_Nurnberg(TPsed,Atherm) # [tons-P yr-1]
            Premob_Carter=compute_remobilization_Carter(TPepi[kt],Atherm) # [tons-P yr-1]
            Premob_methods[:,kt]=np.array([Premob_Nurnberg,Premob_Carter])
        
            
            if method_remob=="Nurnberg":
                Premob[kt]=Premob_Nurnberg
            elif method_remob=="Hanson":
                Premob[kt]=Premob_Hanson
            elif method_remob=="Carter":
                Premob[kt]=Premob_Carter
            else:
                #Premob[kt]=np.nanmean(np.array([Premob_Hanson,Premob_Nurnberg,Premob_Carter]))
                Premob[kt]=np.nanmean(np.array([Premob_Nurnberg,Premob_Carter]))
            #Premob_range[:,kt]=np.array([np.nanmin([Premob_Hanson,Premob_Nurnberg,Premob_Carter]),np.nanmax([Premob_Hanson,Premob_Nurnberg,Premob_Carter])])
            Premob_range[:,kt]=np.array([np.nanmin(Premob_methods[:,kt]),np.nanmax(Premob_methods[:,kt])])
        else:
            Premob[kt]=0
            
            
        # 4. Net sedimentation
        if method_sed=="Vollenweider":
            TP_lake=TPepi[kt]*Vepi/V_lake+TPhypo[kt]*(V_lake-Vepi)/V_lake
            if np.isnan(sigma_max): # Value not specified
                if np.isnan(TPcrit) or np.isnan(P_NS_max):
                    sigma_max=k_sigma/z_mean # [yr-1]
            Pnet_sed[kt],sigma_max_new,TPcrit_new=compute_net_sed_Vollenweider(TP_lake,sigma_max,V_lake,P_NS_max,TPcrit)
            if show_output:
                print("{}: TP_lake={:.2f} mg/m3, P_NS={:.2f} tons/yr, sigma_max={:.2f} yr-1, TPcrit={:.2f} mg/m3".format(datetime.fromtimestamp(tnum_predict[kt],UTC),
                                                                                                     TP_lake,Pnet_sed[kt],sigma_max_new,TPcrit_new))
            param["sigma_max"][kt]=sigma_max_new
            param["TPcrit"][kt]=TPcrit_new
            param["P_NS_max"][kt]=P_NS_max
        elif method_sed=="Hanson" and list(Tepi):
            Pgross=compute_gross_sed_Hanson(Pin[kt],TPepi[kt],Tepi[kt],Vepi,C_pp=0.5) # [tons-P/yr]
            if Pgross>Premob[kt]:
                Pnet_sed[kt]=Pgross-Premob[kt]
            else:
                Pnet_sed[kt]=0
        else:
            raise Exception("Wrong sedimentation method")
        
        
        # 5. Phosphorus transfer due to thermocline vertical motion
        Vepi_new=np.nansum(V_hypso[z_hypso<=hepi[kt+1]])
        dV_epi=Vepi_new-Vepi # [m3]
        if dV_epi>0: # Thermocline deepening
            dmass=dV_epi*TPhypo[kt] # > 0, [mg]
        else: # Thermocline rise (or zero)
            dmass=dV_epi*TPepi[kt] # < 0, [mg]
        
        TPepi[kt+1]=(TPepi[kt]*Vepi+dmass+1e9/(86400*365)*delta_t*(Pin[kt]-Pout_epi[kt]-Premob[kt]-Pnet_sed[kt]+Pz[kt]))/Vepi_new # [mg.m-3]
        TPepi_range[:,kt+1]=np.array([(TPepi[kt]*Vepi+dmass+1e9/(86400*365)*delta_t*(Pin[kt]-Pout_epi[kt]-Premob_range[1,kt]-Pnet_sed[kt]+Pz[kt]))/Vepi_new,\
                              (TPepi[kt]*Vepi+dmass+1e9/(86400*365)*delta_t*(Pin[kt]-Pout_epi[kt]-Premob_range[0,kt]-Pnet_sed[kt]+Pz[kt]))/Vepi_new]) # [mg.m-3]
        
        
        
        
        if TPepi[kt+1]<0:
            TPepi[kt+1]=0
        TPepi_range[TPepi_range[:,kt+1]<0,kt+1]=0
        # Note:
            # If fully mixed: Pz=0, Premob=0 (because always oxic) -> only Pin, Pout and Pnet_sed
            # If stratified but oxic: Premob=0
        
        if hepi[kt+1]<np.max(z_hypso): # Stratified
            # If oxic: TPhypo is only driven by Pz because gross sedimentation=net sedimentation
            TPhypo[kt+1]=(TPhypo[kt]*(V_lake-Vepi)-dmass+1e9/(86400*365)*delta_t*(Premob[kt]-Pz[kt]-Pout_hypo[kt]))/(V_lake-Vepi_new) # [mg.m-3]
            TPhypo_range[:,kt+1]=np.array([(TPhypo[kt]*(V_lake-Vepi)-dmass+1e9/(86400*365)*delta_t*(Premob_range[0,kt]-Pz[kt]-Pout_hypo[kt]))/(V_lake-Vepi_new),\
                                            (TPhypo[kt]*(V_lake-Vepi)-dmass+1e9/(86400*365)*delta_t*(Premob_range[1,kt]-Pz[kt]-Pout_hypo[kt]))/(V_lake-Vepi_new)]) # [mg.m-3]
            
            
        else: # Fully mixed
            TPhypo[kt+1]=TPepi[kt+1]
            TPhypo_range[:,kt+1]=TPepi_range[:,kt+1]
            
        if TPhypo[kt+1]<0:
            TPhypo[kt+1]=0
        TPhypo_range[TPhypo_range[:,kt+1]<0,kt+1]=0
            
    P_fluxes={"Pin":Pin,"Pout_epi":Pout_epi,"Pout_hypo":Pout_hypo,"Premob":Premob,
              "Pnet_sed":Pnet_sed,"Pz":Pz,"Premob_Nurnberg":Premob_methods[0,:],"Premob_Carter":Premob_methods[1,:]}
    
    return tnum_predict,TPepi,TPepi_range,TPhypo,TPhypo_range, P_fluxes, param

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def sensitivity_model(param_name,param_changes,tnum_budget,bool_anoxic,Pin,Qout,hepi,TPepi0,TPhypo0,TPsed,z_hypso,A_hypso,Thypo=np.array([]),Tepi=np.array([]),sigma_max=np.nan,
                     P_NS_max=np.nan,TPcrit=np.nan,k_sigma=10,Kz=1e-7,zout=0,method_sed="Vollenweider",method_remob="average",cst_input=True,show_output=False):
     
    """Function sensitivity_model

    Runs a sensitivity analysis of a specified parameter

    Inputs:
        param_name (string): name of the parameter for which the sensitivity analysis must be performed (options are "Pin","Qout","sigma_max","P_NS_max","TPcrit" and "TPsed")
        param_changes (numpy array (c,) of floats): multiplication factor of the parameter
        tnum_budget (numpy array (n,) of floats): timestamps with river data Pin and Qout (number of seconds since 01.01.1970)
        bool_anoxic (numpy array (n,) of booleans): =True during anoxic period, =False otherwise
        Pin (numpy array (n,) of floats): input P load from rivers computed as Pin=Qin*TPin [tons-P/yr]
        Qout (numpy array (n,) of floats): outflow set equal to Qin by assuming that there is no change of water level [m3/s]
        hepi (numpy array (n,) of floats): real thermocline depth [m]
        TPhypo0 (float): initial TP concentration in the hypolimnion [mg-P/m3]
        TPepi0 (float): initial TP concentration in the epilimnion [mg-P/m3]
        TPsed (float): phosphorus concentration at the sediment surface [mg-P/g-sed] 
        z_hypso (numpy array (m,) of floats): depth values where the lake area is provided [m] 
        A_hypso (numpy array (m,) of floats): lake area at the depth values z_hypso [m2]
        Thypo (numpy array (n,) of floats): hypolimnion temperature [°C], if nan the remobilization rate is not estimated with the Hanson method
        Tepi (numpy array (n,) of floats): epilimnion temperature [°C], if nan the gross sedimentation rate cannot be estimated with the Hanson method
        sigma_max (float): maximal net sedimentation rate reached for low TP concentrations (slope of the linear relationship P_NS=f(TP)) [yr-1]. If nan, estimated from z_mean.
        P_NS_max (float): maximal net sedimentation flux, reached for high TP concentrations [tons-P/yr]. If nan, only linear relationship is used.
        TPcrit (float): critical TP concentration above which sigma is not constant [mg-P/m3]
        k_sigma (float): empirical coefficient to compute sigma_max as sigma_max=k_sigma/z_mean [m.yr-1], common range is 8-16 (Müller et al., 2014) 
        Kz (float): vertical turbulent diffusivity [m2/s] 
        zout (float): depth of the outflow [m]
        method_sed (string): method to compute sedimentation, options are "Hanson" (gross sedimentation) or "Vollenweider" (net sedimentation)
        method_remob (string): method to compute remobilization, options are "Nurnberg", "Hanson", "Carter" or "average"
        cst_input (boolean): =True to set Pin and Qout constant during the simulation period
        show_output (boolean): =True to display sedimentation parameters values at each iteration
    
    Outputs:
        param_test (numpy array (c,) of floats): values of the tested parameter
        TPepi_sens (numpy array (c,n) of floats): modelled TP concentration in the epilimnion as a function of time for all the values of the tested parameters [mg-P/m3]
        TPhypo_sens (numpy array (c,n) of floats): modelled TP concentration in the hypolimnion as a function of time for all the values of the tested parameters [mg-P/m3]
        TPlake_sens (numpy array (c,n) of floats): modelled lake-averaged TP concentration as a function of time for all the values of the tested parameters [mg-P/m3]
    """  
    
    param_dict={"Pin":Pin,"Qout":Qout,"sigma_max":sigma_max,"P_NS_max":P_NS_max,"TPcrit":TPcrit,"TPsed":TPsed}
    param_list=list(param_dict.keys())
    param_test=np.nanmean(param_dict[param_name])*param_changes
    param_val=dict()

    TPepi_sens=np.full((len(param_test),len(tnum_budget)),np.nan)
    TPhypo_sens=np.full((len(param_test),len(tnum_budget)),np.nan)
    TPlake_sens=np.full((len(param_test),len(tnum_budget)),np.nan)
    
    for k in range(len(param_test)):
        print("Iteration {}/{}".format(k+1,len(param_test)))
        
        # Parameters values:
        for kparam in range(len(param_list)):
            if param_name==param_list[kparam]: # This is the parameter to modify
                param_val[param_list[kparam]]=param_test[k]
            else:
                if isinstance(param_dict[param_list[kparam]], np.ndarray):
                    param_val[param_list[kparam]]=np.nanmean(param_dict[param_list[kparam]]) 
                else:
                    param_val[param_list[kparam]]=param_dict[param_list[kparam]]
        
        if cst_input:
            Pin_series=np.full(Pin.shape,param_val["Pin"])
            Qout_series=np.full(Qout.shape,param_val["Qout"])
        else:
            if param_name=="Pin":
                Pin_series=Pin*param_changes[k]
                Qout_series=Qout
            elif param_name=="Qout":
                Pin_series=Pin
                Qout_series=Qout*param_changes[k]
            else:
                Pin_series=Pin
                Qout_series=Qout
            
        print("Pin={}, sigma_max={}, PNSmax={}".format(param_val["Pin"],param_val["sigma_max"],param_val["P_NS_max"]))
        tnum_sens,TPepi_sens[k,:],_,TPhypo_sens[k,:],_,_,_=predict_TP_model(tnum_budget,bool_anoxic,Pin_series,Qout_series,hepi,TPepi0,TPhypo0,param_val["TPsed"],
                                                                                           z_hypso,A_hypso,Thypo,Tepi,sigma_max=param_val["sigma_max"],P_NS_max=param_val["P_NS_max"],TPcrit=param_val["TPcrit"],
                                                                                           k_sigma=k_sigma,Kz=Kz,zout=zout,method_sed=method_sed,method_remob=method_remob,show_output=show_output)
        TPlake_sens[k,:]=np.nanmean(np.array([TPepi_sens[k,:],TPhypo_sens[k,:]]),axis=0)
    
    return param_test,TPepi_sens, TPhypo_sens, TPlake_sens