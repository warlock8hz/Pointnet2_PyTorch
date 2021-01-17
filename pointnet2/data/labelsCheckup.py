def returnIndoor3DLabels(label):
    if label == 0:
        return 'ceiling'
    elif label == 1:
        return 'floor'
    elif label == 2:
        return 'wall'
    elif label == 3:
        return 'column'
    elif label == 4:
        return 'beam'
    elif label == 5:
        return 'window'
    elif label == 6:
        return 'door'
    elif label == 7:
        return 'table'
    elif label == 8:
        return 'chair'
    elif label == 9:
        return 'bookcase'
    elif label == 10:
        return 'sofa'
    elif label == 11:
        return 'board'
    elif label == 12:
        return 'clutter'
    elif label == 13:
        return 'pipe'
    elif label == 14:
        return 'bend-box'
    elif label == 15:
        return 'valve'
    elif label == 16:
        return '90d-elbow'
    elif label == 17:
        return '22.5d-elbow'
    elif label == 18:
        return '45d-elbow'
    elif label == 19:
        return 'tee'
    else: # should not happen
        return 'unknownError'
    return 'unknown' # if error

def returnIndoor3DLabelIds(strName):
    if strName.lower() == 'ceiling':
        return 0
    elif strName.lower() == 'floor':
        return 1
    elif strName.lower() == 'wall':
        return 2
    elif strName.lower() == 'column':
        return 3
    elif strName.lower() == 'beam':
        return 4
    elif strName.lower() == 'window':
        return 5
    elif strName.lower() == 'door':
        return 6
    elif strName.lower() == 'table':
        return 7
    elif strName.lower() == 'chair':
        return 8
    elif strName.lower() == 'bookcase':
        return 9
    elif strName.lower() == 'sofa':
        return 10
    elif strName.lower() == 'board':
        return 11
    elif strName.lower() == 'clutter':
        return 12
    elif strName.lower() == 'pipe':
        return 13
    elif strName.lower() == 'bend-box':
        return 14
    elif strName.lower() == 'valve':
        return 15
    elif strName.lower() == '90d-elbow':
        return 16
    elif strName.lower() == '22.5d-elbow':
        return 17
    elif strName.lower() == '45d-elbow':
        return 18
    elif strName.lower() == 'tee':
        return 19
    else: # should not happen
        return 'unknownError'
    return 255#should be error