

for hpos in "1 9" "2 8" "3 7"
do
read i j <<< $hpos
echo "Hello $i $j"
echo "Mult $(($i * $j))"
done