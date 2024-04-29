
#echo "${0##*x/}"
#echo "${0}" | awk -F'/' '{for(i=1; i<=NF; i++) if($i ~ /^Exp/) print $i}'

current_folder=$(echo "${0}" | awk -F'/' '{for(i=1; i<=NF; i++) if($i ~ /^Exp/) print $i}')
echo $current_folder
echo "hello/$current_folder"
seq_len=336
echo $(expr $seq_len / 2)

for i in "1" \
         "2"; do
    echo $i
done


