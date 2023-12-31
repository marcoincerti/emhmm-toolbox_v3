import numpy as np
import pandas as pd

def read_xls_fixations(xlsname):
    
    print(f'Reading {xlsname}')
    df = pd.read_excel(xlsname)
    
    # Get the headers
    headers = df.columns.tolist()

    # Find the header indices
    SID = headers.index('SubjectID')
    TID = headers.index('TrialID')
    FX = headers.index('FixX')
    FY = headers.index('FixY')
    FD = headers.index('FixD') if 'FixD' in headers else None

    if SID == -1:
        raise ValueError('Error with SubjectID')
    print(f'- found SubjectID in column {SID + 1}')

    if TID == -1:
        raise ValueError('Error with TrialID')
    print(f'- found TrialID in column {TID + 1}')

    if FX == -1:
        raise ValueError('Error with FixX')
    print(f'- found FixX in column {FX + 1}')

    if FY == -1:
        raise ValueError('Error with FixY')
    print(f'- found FixY in column {FY + 1}')

    if FD is not None:
        print(f'- found FixD in column {FD + 1}')

    # Initialize names and trial names
    sid_names = []
    sid_trials = []
    data = []

    # Read data
    for i in range(len(df)):
        mysid = df.iloc[i, SID]
        mytid = df.iloc[i, TID]
        myfxy = [df.iloc[i, FX], df.iloc[i, FY]]

        if FD is not None:
            myfxy.append(df.iloc[i, FD])

        # Convert to string if necessary
        if np.issubdtype(type(mysid), np.number):
            mysid = str(int(mysid))

        if np.issubdtype(type(mytid), np.number):
            mytid = str(int(mytid))

        # Find subject
        s = sid_names.index(mysid) if mysid in sid_names else -1
        if s == -1:
            # New subject
            sid_names.append(mysid)
            sid_trials.append([])
            data.append([])

        # Find trial
        t = -1
        if s != -1:
            t = sid_trials[s].index(mytid) if mytid in sid_trials[s] else -1
        if t == -1:
            sid_trials[s].append(mytid)
            data[s].append([])

        # Put fixation
        data[s][t].append(myfxy)


    print(f'- found {len(sid_names)} subjects:')
    print(' '.join(sid_names))

    for i, subject in enumerate(data):
        print(f'  * subject {i + 1} had {len(subject)} trials')
    
    return data, sid_names, sid_trials