import subprocess
'''
python kmeans.py csv/iris.csv 3
python kmeans.py csv/4clusters.csv 3
python kmeans.py csv/mammal_milk.csv 2
python kmeans.py csv/planets.csv 4
python kmeans.py csv/AccidentsSet03.csv 2
python dbscan.py csv/iris.csv 0.4 3
python dbscan.py csv/4clusters.csv 0.2 3
python dbscan.py csv/mammal_milk.csv 0.2 3
python dbscan.py csv/planets.csv 0.38 7
python dbscan.py csv/AccidentsSet03.csv 0.6 3
python hclustering.py csv/iris.csv 31
python hclustering.py csv/4clusters.csv 8
python hclustering.py csv/mammal_milk.csv 6
python hclustering.py csv/planets.csv 9
python hclustering.py csv/AccidentsSet03.csv 16
'''
kmeans_hyp = [("iris.csv", 3),
            ("4cluster.csv", 3),
            ("mammal_milk.csv", 2),
            ("planets.csv", 4),
            ("AccidentsSet03.csv", 2)]
appendix = ""
for csv, k in kmeans_hyp:
    # print `python kmeans.py csv/<csv> <k>` 
    appendix += f"python kmeans.py csv/{csv} {k}\n"

dbscan_hyp = [("iris.csv", 0.4, 3),
            ("4cluster.csv", 0.2, 3),
            ("mammal_milk.csv", 0.2, 3),
            ("planets.csv", 0.38, 7),
            ("AccidentsSet03.csv", 0.6, 3)]
for csv, eps, minpts in dbscan_hyp:
    # print `python dbscan.py csv/<csv> <eps> <minpts>` 
    appendix += f"python dbscan.py csv/{csv} {eps} {minpts}\n"

agg_hyp = [("iris.csv", 31),
        ("4cluster.csv", 8),
        ("mammal_milk.csv", 6),
        ("planets.csv", 9),
        ("AccidentsSet03.csv", 16)]
for csv, k in agg_hyp:
    # print `python hclustering.py csv/<csv> --k <k>` 
    appendix += f"python hclustering.py csv/{csv} {k}\n"

print(appendix)