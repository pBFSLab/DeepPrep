#! /bin/tcsh -fe

# USAGE: source build_regressors.csh subjId bldrun path/to/bold/run path/to/movement path/to/fcmri
# Files compiled in the path/to/movement and path/to/fcmri directories.

set base_path = `pwd`
set subjId = $1
set bldRun3digit = $2
set bold_path = $base_path/$3
set movement_path = $base_path/$4
set fcmri_path = $base_path/$5

pushd `dirname $0`

awk '{print $2 "  " $3 "  " $4 "  " $5 "  " $6 "  " $7}' \
    ${bold_path}/${bldRun3digit}/${subjId}_bld_rest_reorient_skip_faln_mc.mcdat > \
    ${movement_path}/${subjId}_bld${bldRun3digit}_rest_reorient_skip_faln_mc.par
echo "Generating dat file for ${subjId} - ${bldRun3digit}"
cat ${movement_path}/${subjId}_bld${bldRun3digit}_rest_reorient_skip_faln_mc.par | \
    awk -f par_to_dat.awk > \
    ${movement_path}/${subjId}_bld${bldRun3digit}_rest_reorient_skip_faln_mc.dat
echo "Generating ddat file for ${subjId} - ${bldRun3digit}"
cat ${movement_path}/${subjId}_bld${bldRun3digit}_rest_reorient_skip_faln_mc.par | \
    awk -f par_to_ddat.awk > \
    ${movement_path}/${subjId}_bld${bldRun3digit}_rest_reorient_skip_faln_mc.ddat
echo "Generating rdat file for ${subjId} - ${bldRun3digit}"
cat ${movement_path}/${subjId}_bld${bldRun3digit}_rest_reorient_skip_faln_mc.par | \
    awk -f par_to_rdat.awk > \
    ${movement_path}/${subjId}_bld${bldRun3digit}_rest_reorient_skip_faln_mc.rdat


gawk '$1!~/#/{for (i = 2; i <= 7; i++) printf ("%10s", $i); printf ( "\n" );}' \
    ${movement_path}/${subjId}_bld${bldRun3digit}_rest_reorient_skip_faln_mc.rdat >! \
    ${movement_path}/${subjId}_bld${bldRun3digit}_rest_reorient_skip_faln_mc.rdat.awktemp

gawk '$1!~/#/{for (i = 2; i <= 7; i++) printf ("%10s", $i); printf ( "\n" );}' \
    ${movement_path}/${subjId}_bld${bldRun3digit}_rest_reorient_skip_faln_mc.ddat >! \
    ${movement_path}/${subjId}_bld${bldRun3digit}_rest_reorient_skip_faln_mc.ddat.awktemp

paste ${movement_path}/${subjId}_bld${bldRun3digit}_rest_reorient_skip_faln_mc.rdat.awktemp \
    ${movement_path}/${subjId}_bld${bldRun3digit}_rest_reorient_skip_faln_mc.ddat.awktemp >! \
    ${movement_path}/${subjId}_bld${bldRun3digit}_rest_reorient_skip_faln_mc.rddat

cat ${movement_path}/${subjId}_bld${bldRun3digit}_rest_reorient_skip_faln_mc.rddat | \
    gawk -f trendout.awk >> ${fcmri_path}/${subjId}_mov_regressor.dat

rm ${movement_path}/${subjId}_bld${bldRun3digit}_rest_reorient_skip_faln_mc.rdat.awktemp
rm ${movement_path}/${subjId}_bld${bldRun3digit}_rest_reorient_skip_faln_mc.ddat.awktemp

popd
