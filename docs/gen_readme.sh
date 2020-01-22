jupyter nbconvert --to notebook --inplace --execute README.ipynb
jupyter nbconvert --to markdown README.ipynb
sed 's/README_files/imgs/g' README.md > ../README.md
sed -i 's|README_files|../imgs|g' README.md
cp README_files/* ../imgs/
rm -rf README_files
