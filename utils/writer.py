

def save_prediction(ids, prediction, filename = None):
    lines = ["%s %d" % (ids[i], prediction[0, i]) for i in range(len(ids))]
    with open(filename, 'w') as f:
        f.writelines(lines)
