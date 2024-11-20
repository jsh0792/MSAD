from sksurv.metrics import concordance_index_censored

def c_index(all_censorships, all_event_times, all_risk_scores):
    return concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

def p_value():
    pass