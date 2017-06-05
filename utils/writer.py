

def save_prediction(ids, prediction, filename = None):
    lines = ["%s\t%d" % (ids[i], prediction[i]) for i in range(len(ids))]
    with open(filename, 'w') as f:
        f.writelines(lines)
