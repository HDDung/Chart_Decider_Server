import numpy as np
import matplotlib.pyplot as plt
from itertools import chain

from Utilities.utilities import Utilities

X, y = Utilities.load_csv_Json("Json.csv")
print()
X = np.array(X)
y = np.array(y)
charts = np.unique(y)
result = []
median = []
norms = []
def Normlization(X):
    return np.log2(np.linalg.norm(X))
for chart in charts:
    print(chart)
    tmp = []
    norm = []
    for index in range(len(y)):
        if y[index] == chart:
            tmp.append(X[index])
            norm.append(np.linalg.norm(X[index]))
            #norm.append(Normlization(X[index]))
    result.append(np.mean(tmp, axis=0))
    median.append(np.median(tmp, axis=0))
    # plt.figure()
    # plt.boxplot(norm)
    # plt.show()
    norms.append(np.array(norm))
plt.figure()
plt.boxplot([norms[0], norms[1], norms[7], norms[2], norms[8]], 0, '',
            labels=['100 stacked bar', 'Bar', 'Stacked bar',
                    'Column', 'Stacked column'])

plt.savefig('Statistic.pdf')

plt.figure()
plt.boxplot([norms[3], norms[6]], 0, '', labels=['Donut', 'Pie'])

plt.savefig('Pie.pdf')

plt.figure()
plt.boxplot([norms[4], norms[9], norms[10], norms[12]], 0, '', 0,
            labels=['dual timeseries combination', 'timeseries bubble',
                    'timeseries column', 'timeseries stacked column'])
plt.tight_layout()

plt.savefig('Timeseries.pdf')

plt.figure()
plt.boxplot([norms[5]], 0, '', labels=['Heatmap'])

plt.savefig('Heatmap.pdf')

plt.figure()
plt.boxplot([norms[11]], 0, '', labels=['Time series line'])

plt.savefig('line.pdf')
plt.figure()
plt.boxplot([norms[0] , norms[1], norms[2], norms[3], norms[5], norms[6], norms[7], norms[8], norms[9], norms[10],
             norms[12], norms[4], norms[11]],
            0, 'rs', 0,
            labels=[charts[0] , charts[1], charts[2], charts[3], charts[5],
                    charts[6], charts[7], charts[8], charts[9], charts[10],
                    charts[12], charts[4], charts[11]])
plt.tight_layout()
plt.savefig('full.pdf')

plt.figure()
plt.boxplot([norms[0] , norms[1], norms[2], norms[3], norms[5], norms[6], norms[7], norms[8], norms[9], norms[10],
             norms[12], norms[4], norms[11]],
            0, '', 0,
            labels=[charts[0] , charts[1], charts[2], charts[3], charts[5],
                    charts[6], charts[7], charts[8], charts[9], charts[10],
                    charts[12], charts[4], charts[11]])
plt.tight_layout()
plt.savefig('full_not_outliers.pdf')
plt.figure()
plt.boxplot([norms[0], norms[1], norms[7], norms[3], norms[6], norms[2], norms[8] ], 0, '', 0,
            labels=['100 stacked bar', 'Bar', 'Stacked bar', 'Donut', 'Pie',
                    'Column', 'Stacked column'])
plt.tight_layout()
plt.savefig("Statistic2.pdf")

result = np.array(result)
median = np.array(median)

array2D = []
for index in range(len(result)):
    tmp = []
    for index2 in range(len(result)):
        tmp.append(np.linalg.norm(result[index] - result[index2]))
    array2D.append(tmp)

array2D = np.array(array2D)
avgDist = []
for index in range(len(result)):
    avgDist.append(np.mean(array2D[index]))

avgDist = np.array(avgDist)
np.savetxt("foo.csv", np.around(array2D, decimals=2), delimiter=",", fmt='%1.3f')

np.savetxt("foo1.csv", np.around(avgDist, decimals=2), delimiter=",", fmt='%1.3f')