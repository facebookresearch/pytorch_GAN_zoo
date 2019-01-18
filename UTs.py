import importlib

if __name__ == "__main__":

    UTList = ["ac_criterion"]

    nCorrect = 0
    nFail = 0

    for item in UTList:
        module = importlib.import_module("models.UTs." + item)
        results = module.test()

        if results:
            print(item + " ...OK")
        else:
            print(item + " ...Fail")

        nCorrect += int(results)
        nFail += (1 - int(results))

    print("%d corrects, %d fails" % (nCorrect, nFail))
