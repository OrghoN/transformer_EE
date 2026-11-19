#!/bin/bash

usage ()
{
    cat <<EOF
USAGE: merge_csv.sh DST SRC...

Merge and xz compress multiple csv files together.
NOTE: This script will ignore rows with nan values!
NOTE: All SRC files must have identical header lines (same columns, same order);
      otherwise the script exits with an error and writes nothing.
EOF
    exit 1
}

die ()
{
    echo "$@"
    exit 1
}

check_headers ()
{
    # The merge keeps only the first file's header and concatenates rows by position,
    # so every source must have exactly the same header (same names, same order).
    local src=( "${@}" )
    local ref cur f bad=0

    for f in "${src[@]}"
    do
        [[ -r "${f}" ]] || die "ERROR: cannot read source file: ${f}"
    done

    ref="$(head -n 1 "${src[0]}")"
    [[ -n "${ref}" ]] || die "ERROR: empty header in ${src[0]}"

    for f in "${src[@]:1}"
    do
        cur="$(head -n 1 "${f}")"
        if [[ "${cur}" != "${ref}" ]]
        then
            bad=1
            echo "ERROR: header of ${f} does not match header of ${src[0]}" >&2
            # Report the first column where they differ.
            awk -v a="${ref}" -v b="${cur}" 'BEGIN {
                na = split(a, A, ","); nb = split(b, B, ",")
                n = (na > nb) ? na : nb
                for (i = 1; i <= n; i++)
                    if (A[i] != B[i]) {
                        printf "       first difference at column %d: \"%s\" vs \"%s\" (%d vs %d columns)\n", i, A[i], B[i], na, nb
                        exit
                    }
            }' >&2
        fi
    done

    [[ "${bad}" -eq 0 ]] || die "Mismatched headers. Nothing was written."
}

data_generator ()
{
    local src=( "${@}" )
    head -n 1 "${src[0]}"

    for (( i = 0; i < "${#src[@]}"; i++ ))
    do
        echo -n -e "[$((i+1)) / ${#src[@]}] Processing file ${src[$i]}\r" >&2
        tail -n +2 "${src[$i]}" | grep -v nan
    done
}

dst="${1}"
shift
src=( "${@}" )

[[ "${#src[@]}" -eq 0 ]] && usage
[[ -f "${dst}" ]] && die "Destination file exist already. Will not overwrite."
check_headers "${src[@]}"

data_generator "${src[@]}" | xz -9 -T 0 > "${dst}"

echo -e "\nDONE"

