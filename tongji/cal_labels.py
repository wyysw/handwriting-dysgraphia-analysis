import csv
from io import StringIO

data = """sample_id,game,label
s1,sym,0
s2,sym,1
s3,sym,1
s4,sym,1
s5,sym,0
s6,sym,0
s7,sym,1
s8,sym,0
c1,circle,0
c2,circle,1
c3,circle,1
c4,circle,1
c5,circle,1
c6,circle,1
c7,circle,1
c8,circle,0
c9,circle,0
c10,circle,1
c11,circle,0
c12,circle,1
c13,circle,1
c14,circle,1
c15,circle,1
c16,circle,0
c17,circle,1
c18,circle,1
c19,circle,1
c20,circle,1
c21,circle,1
l1,maze,0
l2,maze,1
l3,maze,1
l4,maze,0
l5,maze,1
l6,maze,1
l7,maze,0
l8,maze,0
l9,maze,0
l10,maze,1
l11,maze,1
l12,maze,1
l13,maze,1
l14,maze,1
l15,maze,1
l16,maze,1
l17,maze,1
l18,maze,1
l19,maze,1
l20,maze,1
l21,maze,1
l22,maze,1
l23,maze,0
l24,maze,1
l25,maze,1
l26,maze,1
l27,maze,0
l28,maze,1
l29,maze,1
l30,maze,0
l31,maze,0
l32,maze,0
l33,maze,1
l34,maze,0"""

stats = {}
reader = csv.DictReader(StringIO(data))
for row in reader:
    game = row['game']
    label = int(row['label'])
    
    if game not in stats:
        stats[game] = {'0': 0, '1': 0}
    stats[game][str(label)] += 1

print("按game类型分类统计：\n")
for game in sorted(stats.keys()):
    count_0 = stats[game]['0']
    count_1 = stats[game]['1']
    total = count_0 + count_1
    print(f"{game:10} -> 0: {count_0:2}  1: {count_1:2}  (总计: {total})")