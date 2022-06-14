
# Reader for UAI .evid files
class EvidReader():
    def __init__(self, filenamepath):
        words = open(filenamepath).read()[:-1].split(' ')
        self.evidence = {}
        num_vars = int(words[0])
        index = 1
        for i in range(num_vars):
            temp = 'var_' + words[index]
            val_temp = int(words[index + 1])
            self.evidence[temp] = val_temp
            index += 2

    def get_evidence(self):
        return self.evidence
