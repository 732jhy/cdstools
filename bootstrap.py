import numpy as np
import scipy.optimize as optim
import scipy.interpolate as polate

def CDS_bootstrap(cds_spreads, yield_curve, cds_tenor, yield_tenor, prem_per_year, R):
    '''
    Bootstraps a credit curve from CDS spreads of varying maturities. Returns the hazard 
    rate values and survival probabilities corresponding to the CDS maturities.

    Args:
        cds_spreads :   vector of CDS spreads
        yield_curve :   vector of risk-free bond yields
        cds_tenor :     vector of maturities corresponding to the given CDS spreads
        yield_tenor :   vector of risk-free bond yield tenor matching yield_curve
        prem_per_year : premiums paid per year on the CDS (i.e. annualy=1, semiannually=2, quarterly=4, monthly=12) 
        R :             recovery rate
    '''

    # Checks
    if len(cds_spreads) != len(cds_tenor):
        print("CDS spread array does match CDS tenor array.")
        return None

    if len(yield_curve) != len(yield_tenor):
        print("Yield curve array does not match yield tenor.")
        return None
    
    # Interpolation/Extrapolation function  
    interp = polate.interp1d(yield_tenor, yield_curve,'linear', fill_value='extrapolate')
    
    # The bootstrap function
    def bootstrap(h, given_haz, s, cds_tenor, yield_curve, prem_per_year, R):
        '''
        Returns the difference between values of payment leg and default leg.
        '''
        a = 1/prem_per_year
        maturities = [0] + list(cds_tenor)    
        pmnt = 0;        dflt = 0;        auc = 0
        
        # 1. Calculate value of payments for given hazard rate curve values
        for i in range(1, len(maturities)-1):
            num_points = int((maturities[i]-maturities[i-1])*prem_per_year + 1)
            t = np.linspace(maturities[i-1], maturities[i], num_points) 
            r = interp(t)
            
            for j in range(1, len(t)):
                surv_prob_prev = np.exp(-given_haz[i-1]*(t[j-1]-t[0]) - auc)
                surv_prob_curr = np.exp(-given_haz[i-1]*(t[j]-t[0]) - auc)
                pmnt += s*a*np.exp(-r[j]*t[j])*0.5*(surv_prob_prev + surv_prob_curr)
                dflt += np.exp(-r[j]*t[j])*(1-R)*(surv_prob_prev - surv_prob_curr)
        
            auc += (t[-1] - t[0])*given_haz[i-1]
        
        # 2. Set up calculations for payments with the unknown hazard rate value
        num_points = int((maturities[-1]-maturities[-2])*prem_per_year + 1)
        t = np.linspace(maturities[-2], maturities[-1], num_points)
        r = interp(t)
        
        for i in range(1, len(t)):
            surv_prob_prev = np.exp(-h*(t[i-1]-t[0]) - auc)
            surv_prob_curr = np.exp(-h*(t[i]-t[0]) - auc)          
            pmnt += s*a*np.exp(-r[i]*t[i])*0.5*(surv_prob_prev + surv_prob_curr)
            dflt += np.exp(-r[i]*t[i])*(1-R)*(surv_prob_prev - surv_prob_curr)
        
        return abs(pmnt-dflt)
    
    haz_rates = []
    surv_prob = []
    t = [0] + list(cds_tenor)
    
    for i in range(len(cds_spreads)):
        get_haz = lambda x: bootstrap(x, haz_rates, cds_spreads[i], cds_tenor[0:i+1], yield_curve[0:i+1], prem_per_year, R)
        haz = round(optim.minimize(get_haz, cds_spreads[i]/(1-R), method='SLSQP', tol = 1e-10).x[0],8)
        cond_surv = (t[i+1]-t[i])*haz
        haz_rates.append(haz)
        surv_prob.append(cond_surv)
    
    return haz_rates, np.exp(-np.cumsum(surv_prob))



def CDS_spread(credit_curve, yield_curve, credit_curve_tenor, yield_tenor, prem_per_year, R, maturity):
    '''
    Returns the spread of a CDS using a yield curve and credit curve

    Args:
        credit_curve :  vector of hazard rates that correspond to CDSs of different maturities
        yield_curve :   vector of yields for risk-free bonds
        credit_curve_tenor :    vector of maturities for CDS contracts corresponding to credit_curve
        yield_tenor :   vector of risk-free bond yield maturities corresponding to yield_curve
        prem_per_year : number of premiums paid per year (i.e. annually=1, semiannually=2, quarterly=4, monthly=12)
        R :             recovery rate
        maturity :      desired CDS maturity
    '''
    # Checks
    if len(yield_curve) != len(yield_tenor):
        print('Yield curve does not match the yield tenor')
        return None
    
    if len(credit_curve) != len(credit_curve_tenor):
        print('Credit curve does not match the credit curve tenor')
        return None            
    
    # I. Get survival probabilities and default probabilities using hazard rate curve
    a = 1/prem_per_year
    num_points = int(credit_curve_tenor[-1]/a + 1)
    t = np.linspace(0, credit_curve_tenor[-1], num_points)
    h = []
    index = 0;  t_index = credit_curve_tenor[index]
    
    for i in range(len(t)):
        if t[i] <= t_index:
            h.append(credit_curve[index])
        else:
            index += 1
            t_index = credit_curve_tenor[index]
            h.append(credit_curve[index])
        
    surv_prob = [1.0]
    
    for i in range(1,len(t)):
        surv_prob.append(a*h[i])
        
    surv_prob = np.exp(-np.cumsum(surv_prob))    
    default_prob = np.asarray([0] + list(-np.diff(surv_prob)))
    
    # II. Interpolate/Extrapolate yield curve values corresponding to payment times and default times    
    interp = polate.interp1d(yield_tenor, yield_curve, 'linear',fill_value='extrapolate')
    pay_periods = np.linspace(0, credit_curve_tenor[-1], num_points)
    mid_periods = np.linspace(a/2, credit_curve_tenor[-1]-a/2, num_points-1)
    yield1 = interp(pay_periods)
    yield2 = interp(mid_periods)
        
    # III. Solve
    PV_pmnt = [np.exp(-yield1[i]*pay_periods[i])*surv_prob[i] for i in range(1,len(pay_periods))] #This works 
    PV_payoff = [(1-R)*default_prob[i+1]*np.exp(-yield2[i]*mid_periods[i]) for i in range(len(mid_periods))]
    PV_accrual = [np.exp(-yield2[i]*mid_periods[i])*0.5*a*default_prob[i+1] for i in range(len(mid_periods))]
    
    return sum(PV_payoff)/(sum(PV_pmnt) + sum(PV_accrual))



def binary_CDS_spread(credit_curve, yield_curve, credit_curve_tenor, yield_tenor, prem_per_year, default_payout, maturity):
    '''
    Returns the spread of a binary CDS using a yield curve and credit curve

    Args:
        credit_curve :  vector of hazard rates that correspond to CDSs of different maturities
        yield_curve :   vector of yields for risk-free bonds
        credit_curve_tenor :    vector of maturities for CDS contracts corresponding to credit_curve
        yield_tenor :   vector of risk-free bond yield maturities corresponding to yield_curve
        prem_per_year : number of premiums paid per year (i.e. annually=1, semiannually=2, quarterly=4, monthly=12)
        default_payout :    amount paid in the event of a default as % of principal
        maturity :      desired CDS maturity
    '''
    # Checks
    if len(yield_curve) != len(yield_tenor):    
        print('Yield curve does not match the yield tenor')
        return None
    
    if len(credit_curve) != len(credit_curve_tenor):
        print('Credit curve does not match the credit curve tenor')
        return None      
    
    # I. Get survival probabilities and default probabilities using hazard rate curve
    a = 1/prem_per_year
    num_points = int(credit_curve_tenor[-1]/a + 1)
    t = np.linspace(0, credit_curve_tenor[-1], num_points)
    h = []
    index = 0;  t_index = credit_curve_tenor[index]

    for i in range(len(t)):
        if t[i] <= t_index:
            h.append(credit_curve[index])
        else:
            index += 1
            t_index = credit_curve_tenor[index]
            h.append(credit_curve[index])
        
    surv_prob = [1.0]
    
    for i in range(1,len(t)):
        surv_prob.append(a*h[i])
        
    surv_prob = np.exp(-np.cumsum(surv_prob))    
    default_prob = np.asarray([0] + list(-np.diff(surv_prob)))    
    
    # II. Interpolate/Extrapolate yield curve values corresponding to payment times and default times    
    interp = polate.interp1d(yield_tenor, yield_curve, 'linear',fill_value='extrapolate')
    pay_periods = np.linspace(0, credit_curve_tenor[-1], num_points)
    mid_periods = np.linspace(a/2, credit_curve_tenor[-1]-a/2, num_points-1)
    yield1 = interp(pay_periods)
    yield2 = interp(mid_periods)
    
    # III. Solve
    PV_pmnt = [np.exp(-yield1[i]*pay_periods[i])*surv_prob[i] for i in range(1,len(pay_periods))]
    PV_payoff = [default_payout*default_prob[i+1]*np.exp(-yield2[i]*mid_periods[i]) for i in range(len(mid_periods))]
    PV_accrual = [np.exp(-yield2[i]*mid_periods[i])*0.5*a*default_prob[i+1] for i in range(len(mid_periods))]
    
    return sum(PV_payoff)/(sum(PV_pmnt) + sum(PV_accrual))

