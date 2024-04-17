
echo "${0##*x/}"
echo "${0}" | awk -F'/' '{for(i=1; i<=NF; i++) if($i ~ /^Exp/) print $i}'

current_folder=$(echo "${0}" | awk -F'/' '{for(i=1; i<=NF; i++) if($i ~ /^Exp/) print $i}')
echo "hello/$current_folder/"



