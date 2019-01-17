class BaseMetric:

    def getScore(self):
        pass

    def updateWithMiniBatch(self, ref, target):
        pass

    def save(self, pathOutput):
        pass
