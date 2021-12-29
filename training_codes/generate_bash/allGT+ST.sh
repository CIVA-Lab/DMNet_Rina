python ../train_unify_allGTST.py all GT+ST mask ../yml/all_eachcell_GTST.yml
python ../train_unify_allGTST.py all GT+ST center ../yml/center_ctc.yml
python ../train_unify_allGTST.py allBF GT+ST mask ../yml/all_eachcell_GTST512.yml
python ../train_unify_allGTST.py allBF GT+ST center ../yml/center_ctc512.yml
python ../train_unify.py 3Dtrain GT+ST mask ../yml/all_eachcell_3DGTST.yml
python ../train_unify.py 3Dtrain GT+ST shapemarker ../yml/all_eachcell_3DGTST.yml
