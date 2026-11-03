call gams merge --simName=GFT       --outDir="." --csvName="GFTMerge" --ifCSVAppend=0
call gams merge --simName=GFTCET    --outDir="." --csvName="GFTMerge" --ifCSVAppend=1
call gams merge --simName=GFTCapFix --outDir="." --csvName="GFTMerge" --ifCSVAppend=1
call gams merge --simName=GFTLnd    --outDir="." --csvName="GFTMerge" --ifCSVAppend=1
call gams merge --simName=GFTFlxFF  --outDir="." --csvName="GFTMerge" --ifCSVAppend=1
