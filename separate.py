'''
Separates the results of testing into separate files.
(real answers, model answers, random answers, testing answers)
'''

path = 'datasets/facebook2/'

realanswers = ''
modelanswers = ''
randomanswers = ''
testinganswers = ''

f = open(path + 'testing.txt', 'r')
for line in f:
    fline = ''
    try:
        fline = line.split(':')[1].strip()
    except:
        continue
    fline = "%s%s" % (fline[0].upper(), fline[1:])
    fline = fline + '.'
    if line.startswith("real answer"):
        realanswers = realanswers + ' ' + fline
    if line.startswith("model answer"):
        modelanswers = modelanswers + ' ' + fline
    if line.startswith("random answer"):
        randomanswers = randomanswers + ' ' + fline
    if line.startswith("testing answer"):
        testinganswers = testinganswers + ' ' + fline
f.close()

f = open(path + 'realanswers.txt', 'w')
f.write(realanswers.strip())
f.close()

f = open(path + 'modelanswers.txt', 'w')
f.write(modelanswers.strip())
f.close()

f = open(path + 'randomanswers.txt', 'w')
f.write(randomanswers.strip())
f.close()

f = open(path + 'testinganswers.txt', 'w')
f.write(testinganswers.strip())
f.close()
